from typing import Union, Type, List, Tuple, Optional

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from pangteen.network import cfg
from pangteen.network.common import helper
from pangteen.network.network_analyzer import NetworkAnalyzer
from pangteen.network.ptnet.conv_blocks import MultiBasicConvBlock, BasicConvBlock, UpSampleBlock, GSC, DownSampleBlock, \
    UXConvBlock, MetaNeXtBlock, InceptionDWConv
from pangteen.network.ptnet.fushion_blocks import SelectiveFusionBlock, MultiScaleFusionBlock
from pangteen.network.ptnet.mamba_blocks import MambaLayer
from pangteen.network.ptnet.nnunet import nnUNet
from pangteen.network.ptnet.ptnet import PangTeenNet
from pangteen.network.ptnet.ptresunet import SkipResBlock, MultiResBlock
from pangteen.network.ptnet.ptukan import EmptyBlock, UKANEncoderBlock, UKANDownSampleBlock, UKANDecoderBlock, \
    UKANUpSampleBlock
from pangteen.network.ptnet.transformer_blocks import TransformerBlock


class SKIM_UNet(PangTeenNet):
    """
    PangTeen: 把 UKAN 模型改造成 3D 版本。
    """

    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 skip_merge_type='add',
                 deep_supervision: bool = False,
                 encoder_types: list = ['Conv', 'Conv', 'Conv', 'Conv', 'Conv', 'Conv'],
                 decoder_types: list = ['Conv', 'Conv', 'Conv', 'Conv', 'Conv', 'Conv'],
                 spatial_dim: int = 3,
                 feature_channels=[32, 64, 128, 256, 320, 320],
                 res_path_count: Optional[List] = None,  # [4, 3, 2, 1],
                 no_kan=False,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 block_depths=1,
                 select_fusion=False,
                 pool: str = 'conv',
                 use_mambav2=False,
                 down_sample_first=False,
                 patch_size=(128, 128, 128),
                 proj_size=[64, 64, 64, 32, 32, 32],
                 mlp_ratio=[4, 3, 3, 2, 1, 1],
                 upsample_last=False,
                 skip_fusion=False,
                 tri_orientation=False,
                 use_max=False,
                 **invalid_args,
                 ):
        """
        Args:
            encoder_types: 编码器的类型，'Conv'表示普通的卷积层，'KAN'表示UKAN的卷积层。
            decoder_types: 解码器的类型，'Conv'表示普通的卷积层，'KAN'表示UKAN的卷积层。
            spatial_dim: 空间维度。
            feature_channels: 每个stage的通道数。
            no_kan: 是否不使用UKAN，改用MLP。
            select_fusion: 是否使用选择性融合。
        """
        super().__init__(n_stages, enable_skip_layer=True, skip_merge_type=skip_merge_type,
                         deep_supervision=deep_supervision,
                         down_sample_first=down_sample_first)

        # 预设的参数。
        self.spatial_dim = spatial_dim
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.encoder_types = encoder_types
        self.decoder_types = decoder_types
        if isinstance(block_depths, int):
            block_depths = [block_depths] * n_stages
        feature_channels = feature_channels[:n_stages]
        proj_size = proj_size[:n_stages]
        assert len(encoder_types) == n_stages, "The number of encoder_types and n_stages must be the same."
        assert len(decoder_types) == n_stages, "The number of decoder_types and n_stages must be the same."
        assert len(block_depths) == n_stages, "The number of block_depths and n_stages must be the same."
        assert res_path_count is None or len(
            res_path_count) == n_stages - 1, "The number of res_path_count must be n_stages - 1."
        assert select_fusion ^ (skip_merge_type is not None), "select_fusion and skip_merge_type must be exclusive."

        self.pool_op = helper.get_matching_pool_op(dimension=self.spatial_dim, pool_type='max')
        self.interpolate = helper.get_matching_interpolate(dimension=self.spatial_dim)
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_stages)]
        concat_multiplier = 2 if skip_merge_type == 'concat' else 1
        input_size = patch_size[0] * patch_size[1] * patch_size[2]

        # 编码器。
        in_chans = input_channels
        for s in range(self.n_stages):
            input_size = input_size // (strides[s][0] * strides[s][1] * strides[s][2])
            if encoder_types[s] == 'KAN':
                # UKANEncoderBlock 包含了 下采样。
                self.encoder_layers.append(UKANEncoderBlock(
                    in_channels=in_chans, out_channels=feature_channels[s],
                    conv_op=conv_op, stride=strides[s],
                    norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                    norm_layer=norm_layer, drop_rate=drop_rate, drop_path_rate=self.dpr[s],
                    no_kan=no_kan
                ))
                self.down_sample_blocks.append(EmptyBlock())
            elif encoder_types[s] == 'MChannel':
                assert self.down_sample_first, "MChannel only supports down_sample_first."
                self.encoder_layers.append(nn.Sequential(*[
                    GSC(feature_channels[s]),
                    *[MambaLayer(input_size, channel_token=True, use_v2=use_mambav2) for _ in range(block_depths[s])]
                ]))
                self.down_sample_blocks.append(DownSampleBlock(
                    conv_op, in_chans, feature_channels[s], kernel_sizes[s], strides[s],
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, pool
                ))
            elif encoder_types[s] == 'MSpatial':
                assert self.down_sample_first, "MSpatial only supports down_sample_first."
                self.encoder_layers.append(nn.Sequential(*[
                    GSC(feature_channels[s]),
                    *[MambaLayer(feature_channels[s], channel_token=False, use_v2=use_mambav2) for _ in
                      range(block_depths[s])]
                ]))
                self.down_sample_blocks.append(DownSampleBlock(
                    conv_op, in_chans, feature_channels[s], kernel_sizes[s], strides[s],
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, pool
                ))
            elif encoder_types[s] == 'Conv':
                self.encoder_layers.append(MultiBasicConvBlock(
                    n_conv_per_stage[s], input_channels=in_chans, output_channels=feature_channels[s],
                    conv_op=conv_op, kernel_size=kernel_sizes[s], stride=1,
                    norm_op=norm_op, norm_op_kwargs=norm_op_kwargs
                ))
                self.down_sample_blocks.append(UKANDownSampleBlock(
                    pool_op=self.pool_op, kernel_size=kernel_sizes[s], stride=strides[s]
                ))
            elif encoder_types[s] == 'GSC':
                self.encoder_layers.append(nn.Sequential(*[
                    GSC(in_chans),
                ]))
                self.down_sample_blocks.append(DownSampleBlock(
                    conv_op, in_chans, feature_channels[s], kernel_sizes[s], strides[s],
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, pool
                ))
            elif encoder_types[s] == 'Res':
                self.encoder_layers.append(MultiResBlock(
                    conv_op, in_chans, feature_channels[s], conv_bias=conv_bias,
                    norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin, nonlin_kwargs=nonlin_kwargs
                ))
                self.down_sample_blocks.append(DownSampleBlock(
                    conv_op, feature_channels[s], feature_channels[s], kernel_sizes[s], strides[s],
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, pool
                ))
            elif encoder_types[s] == 'EPA':
                assert self.down_sample_first, "MChannel only supports down_sample_first."
                self.encoder_layers.append(nn.Sequential(*[
                    TransformerBlock(input_size, feature_channels[s], proj_size[s], 4, pos_embed=True)
                    for _ in range(block_depths[s])]))
                self.down_sample_blocks.append(DownSampleBlock(
                    conv_op, in_chans, feature_channels[s], kernel_sizes[s], strides[s],
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, pool
                ))
            elif encoder_types[s] == 'XT':
                assert self.down_sample_first, "XT only supports down_sample_first."
                self.encoder_layers.append(nn.Sequential(*[MetaNeXtBlock(
                    dim=feature_channels[s],
                    out_dim=feature_channels[s],
                    token_mixer=InceptionDWConv,
                    norm_layer=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    mlp_ratio=mlp_ratio[s],
                    drop_path=self.dpr[s],
                ) for _ in range(n_conv_per_stage[s])]))
                self.down_sample_blocks.append(DownSampleBlock(
                    conv_op, in_chans, feature_channels[s], kernel_sizes[s], strides[s],
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, pool
                ))
            else:
                raise ValueError(f"Unknown encoder type: {encoder_types[s]}")
            in_chans = feature_channels[s]

        # 解码器。
        input_size = patch_size[0] * patch_size[1] * patch_size[2]
        for s in range(self.n_stages - 1):
            input_size = input_size // (strides[s][0] * strides[s][1] * strides[s][2])
            if select_fusion:
                self.skip_merge_blocks.append(
                    SelectiveFusionBlock(feature_channels[s], conv_op, 'mlp', use_max=use_max))

            if res_path_count:
                self.skip_layers.append(nn.Sequential(*[
                    SkipResBlock(conv_op, feature_channels[s], norm_op, norm_op_kwargs, nonlin, nonlin_kwargs) for _
                    in range(res_path_count[s])
                ]))

            if encoder_types[s] == 'KAN':
                # 不支持 'concat'。
                self.decoder_layers.append(UKANDecoderBlock(
                    in_channels=feature_channels[s], conv_op=conv_op,
                    norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                    norm_layer=norm_layer, drop_rate=drop_rate, drop_path_rate=self.dpr[s]
                ))
                # UKANUpSampleBlock 包含了 上采样和卷积。
                self.up_sample_blocks.append(UKANUpSampleBlock(
                    in_channels=feature_channels[s + 1], out_channels=feature_channels[s],
                    conv_op=conv_op, interpolate_mode=self.interpolate, stride=strides[s + 1],
                    norm_op=norm_op, norm_op_kwargs=norm_op_kwargs
                ))
            elif encoder_types[s] == 'MChannel':
                # 不支持 'concat'。
                self.decoder_layers.append(nn.Sequential(*[
                    GSC(feature_channels[s] * concat_multiplier),
                    *[MambaLayer(input_size, channel_token=True, use_v2=use_mambav2) for _ in range(block_depths[s])]
                ]))
                self.up_sample_blocks.append(UpSampleBlock(
                    conv_op, feature_channels[s + 1], feature_channels[s], strides[s + 1], conv_bias
                ))
            elif encoder_types[s] == 'MSpatial':
                self.decoder_layers.append(nn.Sequential(*[
                    GSC(feature_channels[s] * concat_multiplier),
                    *[MambaLayer(feature_channels[s] * concat_multiplier, channel_token=False, use_v2=use_mambav2) for _
                      in range(block_depths[s])]
                ]))
                self.up_sample_blocks.append(UpSampleBlock(
                    conv_op, feature_channels[s + 1], feature_channels[s], strides[s + 1], conv_bias
                ))
            elif encoder_types[s] == 'Conv':
                self.decoder_layers.append(MultiBasicConvBlock(
                    n_conv_per_stage_decoder[s], input_channels=feature_channels[s] * concat_multiplier,
                    output_channels=feature_channels[s],
                    conv_op=conv_op, kernel_size=3, stride=1,
                    norm_op=norm_op, norm_op_kwargs=norm_op_kwargs
                ))
                self.up_sample_blocks.append(UpSampleBlock(
                    conv_op, feature_channels[s + 1], feature_channels[s], strides[s + 1], conv_bias
                ))
            elif encoder_types[s] == 'GSC':
                self.decoder_layers.append(nn.Sequential(*[
                    GSC(feature_channels[s] * concat_multiplier),
                ]))
                self.up_sample_blocks.append(UpSampleBlock(
                    conv_op, feature_channels[s + 1], feature_channels[s], strides[s + 1], conv_bias
                ))
            elif encoder_types[s] == 'Res':
                self.decoder_layers.append(MultiResBlock(
                    conv_op, feature_channels[s] * concat_multiplier, feature_channels[s], conv_bias=conv_bias,
                    norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin, nonlin_kwargs=nonlin_kwargs
                ))
                self.up_sample_blocks.append(UpSampleBlock(
                    conv_op, feature_channels[s + 1], feature_channels[s], strides[s + 1], conv_bias
                ))
            elif encoder_types[s] == 'EPA':
                self.decoder_layers.append(nn.Sequential(*[
                    TransformerBlock(input_size, feature_channels[s], proj_size[s], 4, pos_embed=True)
                    for _ in range(block_depths[s])]))
                self.up_sample_blocks.append(UpSampleBlock(
                    conv_op, feature_channels[s + 1], feature_channels[s], strides[s + 1], conv_bias
                ))
            elif encoder_types[s] == 'XT':
                self.decoder_layers.append(nn.Sequential(*[MetaNeXtBlock(
                    dim=feature_channels[s] * concat_multiplier,
                    out_dim=feature_channels[s],
                    token_mixer=InceptionDWConv,
                    norm_layer=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    mlp_ratio=mlp_ratio[s],
                    drop_path=self.dpr[s],
                ) for _ in range(n_conv_per_stage_decoder[s])]))
                self.up_sample_blocks.append(UpSampleBlock(
                    conv_op, feature_channels[s + 1], feature_channels[s], strides[s + 1], conv_bias
                ))
            else:
                raise ValueError(f"Unknown encoder type: {encoder_types[s]}")

            if upsample_last:
                self.seg_layers.append(nn.Sequential(*[
                    UpSampleBlock(
                        conv_op, feature_channels[s], feature_channels[s], (2, 2, 2), conv_bias
                    ),
                    BasicConvBlock(
                        input_channels=feature_channels[s], output_channels=num_classes,
                        conv_op=conv_op, kernel_size=1, stride=1,
                    ),
                ]))
            else:
                self.seg_layers.append(nn.Sequential(*[
                    BasicConvBlock(
                        input_channels=feature_channels[s], output_channels=num_classes,
                        conv_op=conv_op, kernel_size=1, stride=1,
                    ),
                ]))

            if skip_fusion:
                self.skip_fusion_block = MultiScaleFusionBlock(
                    feature_channels[:-1], conv_op, norm_op, nonlin, nonlin_kwargs, norm_layer,
                    use_mamba_v2=use_mambav2, tri_orientation=tri_orientation
                )

        self.bottle_neck = None


def analyze_mamba_ukan_count(mamba_count=3, kan_count=2):
    type_list = ['MChannel'] * mamba_count + ['KAN'] * kan_count
    network = SKIM_UNet(
        encoder_types=type_list,
        decoder_types=type_list,
        **cfg.stage5_network_args
    ).cuda()

    NetworkAnalyzer(network, print_flops=True, test_backward=True).analyze()


def analyze_xt_count():
    type_list = ['XT'] * 6
    network = SKIM_UNet(
        encoder_types=type_list,
        decoder_types=type_list,
        down_sample_first=True,
        **cfg.default_network_args
    ).cuda()

    NetworkAnalyzer(network, print_flops=True, test_backward=True).analyze()


def analyze_nnunet():
    type_list = ['Conv'] * 6
    network = SKIM_UNet(
        encoder_types=type_list,
        decoder_types=type_list,
        **cfg.default_network_args
    ).cuda()

    NetworkAnalyzer(network, print_flops=True, test_backward=True).analyze()


def analyze_multi_resunet():
    type_list = ['Res'] * 5
    network = SKIM_UNet(
        encoder_types=type_list,
        decoder_types=type_list,
        select_fusion=True,
        skip_merge_type=None,
        res_path_count=[4, 3, 2, 1],
        **cfg.stage5_network_args
    ).cuda()

    NetworkAnalyzer(network, print_flops=True, test_backward=True).analyze()

def analyze_ukan():
    type_list = ['Conv'] * 3 + ['KAN'] * 2
    network = SKIM_UNet(
        encoder_types=type_list,
        decoder_types=type_list,
        **cfg.stage5_network_args
    ).cuda()

    NetworkAnalyzer(network, print_flops=True, test_backward=True).analyze()


def analyze_mine(xt_cnt, kan_cnt):
    type_list = ['XT'] * xt_cnt + ['KAN'] * kan_cnt
    network = SKIM_UNet(
        encoder_types=type_list,
        decoder_types=type_list,
        down_sample_first=True,
        select_fusion=True,
        skip_merge_type=None,
        skip_fusion=True,
        **cfg.default_network_args
    ).cuda()

    NetworkAnalyzer(network, print_flops=True, test_backward=True).analyze()

if __name__ == "__main__":
    # analyze_nnunet()
    # analyze_multi_resunet()
    # analyze_ukan()
    analyze_mine(xt_cnt=3, kan_cnt=3)
    # analyze_mamba_ukan_count(3, 2)
    # analyze_xt_count()
    pass
