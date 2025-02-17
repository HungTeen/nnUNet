import os
from typing import Union, List, Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from pangteen.network.common import helper
from pangteen.network.common.kan import KANLinear
from pangteen.network.km_unet.block import SS2D, EMA
from pangteen.network.ptnet.conv_blocks import BasicConvBlock, MultiBasicConvBlock
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

from pangteen.network.ptnet.fushion_blocks import SelectiveFusionBlock
from pangteen.network.ptnet.ptnet import PangTeenNet
from pangteen.network.ptnet.ukan import PatchEmbed, KANBlock


class UKANDownSampleBlock(nn.Module):

    def __init__(self, pool_op: Type[torch.nn.Module], kernel_size: Union[int, list], stride: Union[int, list]):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding: Union[list, int] = 0
        if isinstance(kernel_size, int):
            self.padding = (kernel_size - 1) // 2
        elif isinstance(kernel_size, list):
            self.padding = [(a - 1) // 2 for a in kernel_size]
        self.pool_op = pool_op(self.kernel_size, stride, padding=self.padding)
        self.nonlin = nn.ReLU()

    def forward(self, x):
        x = self.pool_op(x)
        x = self.nonlin(x)
        return x


class UKANUpSampleBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 conv_op: Type[_ConvNd],
                 interpolate_mode: str,
                 stride: Union[int, list],
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None):
        super().__init__()
        self.interpolate_mode = interpolate_mode
        self.stride = stride
        self.conv_decode_list = MultiBasicConvBlock(
            2,
            input_channels=in_channels, output_channels=out_channels,
            conv_op=conv_op, kernel_size=3, stride=1,
            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            reverse_order=True
        )

    def forward(self, x):
        return F.relu(F.interpolate(self.conv_decode_list(x), scale_factor=self.stride, mode=self.interpolate_mode))

class UKANEncoderBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 conv_op: Type[_ConvNd], stride: Union[int, list],
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 norm_layer=nn.LayerNorm,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 no_kan=False,
                 ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            input_channels=in_channels, output_channels=out_channels,
            conv_op=conv_op, patch_size=3, stride=stride,
        )
        self.kan_encode_blocks = nn.ModuleList([KANBlock(
            in_features=out_channels, conv_op=conv_op,
            drop=drop_rate, drop_path=drop_path_rate, no_kan=no_kan,
            norm_layer=norm_layer, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
        )])
        self.encode_norm = norm_layer(out_channels)

    def forward(self, x):
        B = x.shape[0]
        x, patch = self.patch_embed(x)  # B C H W D -> B, N, C，这里的 N 已经是降采样后的大小了。
        for blk in self.kan_encode_blocks:
            x = blk(x, patch)  # B, N, C
        x = self.encode_norm(x)
        x = x.reshape(B, *patch, -1)  # B, N, C -> B, H, W, D, C
        x = helper.channel_to_the_second(x)  # B, C, H, W, D
        return x


class UKANDecoderBlock(nn.Module):

    def __init__(self, in_channels: int,
                 conv_op: Type[_ConvNd],
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 norm_layer=nn.LayerNorm,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 no_kan=False,
                 ):
        super().__init__()
        self.kan_decode_blocks = nn.ModuleList([KANBlock(
            in_features=in_channels, conv_op=conv_op,
            drop=drop_rate, drop_path=drop_path_rate, no_kan=no_kan,
            norm_layer=norm_layer, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
        )])
        self.decode_norm = norm_layer(in_channels)

    def forward(self, x):
        B = x.shape[0]
        patch = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)  # B C H W D -> B, N, C
        for blk in self.kan_decode_blocks:
            x = blk(x, patch)  # B, N, C
        x = self.decode_norm(x)
        x = x.reshape(B, *patch, -1)  # B, N, C -> B, H, W, D, C
        x = helper.channel_to_the_second(x)  # B, C, H, W, D
        return x


class EmptyBlock(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class UKAN_3D(PangTeenNet):
    """
    PangTeen: 把 UKAN 模型改造成 3D 版本。
    """

    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 encode_kan_num: int = 2,
                 decode_kan_num: int = 2,
                 reverse_kan_order: bool = False,
                 spatial_dim: int = 3,
                 # embed_dims=[32, 64, 256, 320, 512],
                 embed_dims=[32, 64, 128, 256, 512],
                 no_kan=False,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1],
                 **invalid_args,
                 ):
        """
        Args:
            encode_kan_num: 编码器中 KAN 的层数。
            decode_kan_num: 解码器中 KAN 的层数。
            reverse_kan_order: 是否反转 KAN 的顺序，默认为 False，即从最底层开始。
            spatial_dim: 空间维度。
        """
        super().__init__(n_stages, enable_skip_layer=True, skip_merge_type='add', deep_supervision=False, down_sample_first=False)

        # 预设的参数。
        self.spatial_dim = spatial_dim
        self.reverse_kan_order = reverse_kan_order
        self.kernel_sizes = kernel_sizes
        self.strides = strides

        self.pool_op = helper.get_matching_pool_op(dimension=self.spatial_dim, pool_type='max')
        self.interpolate = helper.get_matching_interpolate(dimension=self.spatial_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 卷积，直到特征图数量达到 kan_input_dim。
        in_chans = input_channels
        cnt = 0
        for s in range(self.n_stages):
            if self.is_kan_layer(s, encode_kan_num):
                self.encoder_layers.append(UKANEncoderBlock(
                    in_channels=in_chans, out_channels=embed_dims[s],
                    conv_op=conv_op, stride=strides[s],
                    norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                    norm_layer=norm_layer, drop_rate=drop_rate, drop_path_rate=dpr[cnt],
                    no_kan=no_kan
                ))
                cnt += 1
                self.down_sample_blocks.append(EmptyBlock())
            else:
                self.encoder_layers.append(MultiBasicConvBlock(
                    2, input_channels=in_chans, output_channels=embed_dims[s],
                    conv_op=conv_op, kernel_size=3, stride=1,
                    norm_op=norm_op, norm_op_kwargs=norm_op_kwargs
                ))
                self.down_sample_blocks.append(UKANDownSampleBlock(
                pool_op=self.pool_op, kernel_size=kernel_sizes[s], stride=strides[s]
            ))
            in_chans = embed_dims[s]

        cnt = 0
        for s in range(self.n_stages - 1):
            if self.is_kan_layer(s, decode_kan_num):
                self.decoder_layers.append(UKANDecoderBlock(
                    in_channels=embed_dims[s], conv_op=conv_op,
                    norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                    norm_layer=norm_layer, drop_rate=drop_rate, drop_path_rate=dpr[cnt]
                ))
                cnt += 1
            else:
                self.decoder_layers.append(MultiBasicConvBlock(
                    2, input_channels=embed_dims[s], output_channels=embed_dims[s],
                    conv_op=conv_op, kernel_size=3, stride=1,
                    norm_op=norm_op, norm_op_kwargs=norm_op_kwargs
                ))

            self.up_sample_blocks.append(UKANUpSampleBlock(
                in_channels=embed_dims[s + 1], out_channels=embed_dims[s],
                conv_op=conv_op, interpolate_mode=self.interpolate, stride=strides[s + 1],
                norm_op=norm_op, norm_op_kwargs=norm_op_kwargs
            ))

        self.seg_layers.append(nn.Sequential(*[
            UKANUpSampleBlock(
                in_channels=embed_dims[0], out_channels=embed_dims[0],
                conv_op=conv_op, interpolate_mode=self.interpolate, stride=strides[0],
                norm_op=norm_op, norm_op_kwargs=norm_op_kwargs
            ),
            BasicConvBlock(
                input_channels=embed_dims[0], output_channels=num_classes,
                conv_op=conv_op, kernel_size=1, stride=1,
            ),
        ]))

        self.bottle_neck = None

    def is_kan_layer(self, stage, kan_count):
        if self.reverse_kan_order:
            return stage >= self.n_stages - kan_count
        else:
            return stage < kan_count


class SFUKAN_3D(UKAN_3D):
    """
    PangTeen: 把 UKAN 模型改造成 3D 版本。
    """

    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 encode_kan_num: int = 2,
                 decode_kan_num: int = 2,
                 reverse_kan_order: bool = False,
                 spatial_dim: int = 3,
                 # embed_dims=[32, 64, 256, 320, 512],
                 embed_dims=[32, 64, 128, 256, 512],
                 no_kan=False,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1],
                 **invalid_args,
                 ):
        super().__init__(input_channels, n_stages, conv_op, kernel_sizes, strides, num_classes, norm_op, norm_op_kwargs,
                         dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, encode_kan_num, decode_kan_num,
                         reverse_kan_order, spatial_dim, embed_dims, no_kan, drop_rate, drop_path_rate, norm_layer, depths)
        for s in range(n_stages - 1):
            self.skip_merge_blocks.append(SelectiveFusionBlock(embed_dims[s], conv_op, 'mlp'))


if __name__ == "__main__":
    # 设置CUDA可见设备
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    network = UKAN_3D(
        input_channels=1,
        n_stages=5,
        conv_op=nn.Conv3d,
        kernel_sizes=[3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2],
        num_classes=2,
        norm_op=helper.get_matching_instancenorm(dimension=3),
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'negative_slope': 1e-2, 'inplace': True},
        encode_kan_num=2,
        decode_kan_num=2,
        reverse_kan_order=True,
    ).cuda()

    # 输出网络结构
    print(network)

    x = torch.zeros((2, 1, 64, 128, 128), requires_grad=True).cuda()
    y = torch.rand((2, 2, 64, 128, 128), requires_grad=False).cuda()

    with torch.autocast(device_type='cuda', enabled=True):
        pred = network(x)
        print(pred.size())

        loss = F.cross_entropy(pred, y.argmax(1))
        loss.backward()

        print(loss)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(network))