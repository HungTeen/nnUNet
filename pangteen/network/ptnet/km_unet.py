from typing import Union, List, Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from pangteen.network.common import helper
from pangteen.network.common.kan import KANLinear
from pangteen.network.km_unet.block import SS2D, EMA
from pangteen.network.ptnet.conv_blocks import BasicConvBlock
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

from pangteen.network.ptnet.ptnet import PangTeenNet


class KM_UNet_3D(PangTeenNet):
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
                 kan_layer_num: int = 2,
                 spatial_dim: int = 3,
                 embed_dims=[32, 64, 256, 320, 512],
                 # embed_dims=[32, 64, 128, 256, 512],
                 ss2d_dim=None,  # [16, 32, 128, 160]
                 no_kan=False,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1],
                 **invalid_args,
                 ):
        """
        Args:
            kan_layer_num: KAN 的层数。
            spatial_dim: 空间维度。
        """
        super().__init__(n_stages, enable_skip_layer=True, skip_merge_type='add', deep_supervision=False)

        # 预设的参数。
        self.conv_layer_num = n_stages - kan_layer_num
        self.kan_layer_num = kan_layer_num
        self.spatial_dim = spatial_dim
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.ss2d_enable = ss2d_dim is not None
        self.ema_enable = False

        self.pool_op = helper.get_matching_pool_op(dimension=self.spatial_dim, pool_type='max')
        self.interpolate = helper.get_matching_interpolate(dimension=self.spatial_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        if spatial_dim == 2:
            self.ss2d_class = SS2D
            self.ema_class = EMA
        elif spatial_dim == 3:
            pass
        else:
            raise ValueError(f"Unsupported spatial dimension: {spatial_dim}.")

        # 卷积，直到特征图数量达到 kan_input_dim。
        in_chans, out_chans = input_channels, embed_dims[0]
        for i in range(self.conv_layer_num):
            self.encoder_layers.append(MultiBasicConvBlock(
                2, input_channels=in_chans, output_channels=out_chans,
                conv_op=conv_op, kernel_size=3, stride=1,
                norm_op=norm_op, norm_op_kwargs=norm_op_kwargs
            ))
            self.down_sample_blocks.append(UKANDownSampleBlock(
                pool_op=self.pool_op, kernel_size=kernel_sizes[i], stride=strides[i]
            ))
            self.decoder_layers.append(EmptyDecoderBlock())

            in_chans = out_chans
            out_chans = embed_dims[i + 1]

        self.bottle_neck = nn.Sequential(*[
            UKANEncoderBlock(
                in_channels=embed_dims[-2], out_channels=embed_dims[-1],
                conv_op=conv_op, stride=strides[-1],
                norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                norm_layer=norm_layer, drop_rate=drop_rate, drop_path_rate=drop_path_rate,
                no_kan=no_kan
            ),
            # UKANUpSampleBlock(
            #     in_channels=embed_dims[-1], out_channels=embed_dims[-2],
            #     conv_op=conv_op, interpolate_mode=self.interpolate, stride=strides[-1],
            #     norm_op=norm_op, norm_op_kwargs=norm_op_kwargs
            # ),
            # UKANDecoderBlock(
            #     in_channels=embed_dims[-2], conv_op=conv_op,
            #     norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            #     nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
            #     norm_layer=norm_layer, drop_rate=drop_rate, drop_path_rate=dpr[-1],
            # )
        ])

        for i in range(self.kan_layer_num):
            if i < self.kan_layer_num - 1:
                self.encoder_layers.append(UKANEncoderBlock(
                    in_channels=in_chans, out_channels=out_chans,
                    conv_op=conv_op, stride=strides[i + self.conv_layer_num],
                    norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                    norm_layer=norm_layer, drop_rate=drop_rate, drop_path_rate=drop_path_rate,
                    no_kan=no_kan
                ))
            self.down_sample_blocks.append(EmptyDecoderBlock())
            self.decoder_layers.append(UKANDecoderBlock(
                in_channels=in_chans, conv_op=conv_op,
                norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                norm_layer=norm_layer, drop_rate=drop_rate, drop_path_rate=dpr[i],
            ))

            in_chans = out_chans
            if i < self.kan_layer_num - 1:
                out_chans = embed_dims[i + 1 + self.conv_layer_num]

        for i in range(1, n_stages):
            in_chans, out_chans = embed_dims[i], embed_dims[i - 1]
            self.up_sample_blocks.append(UKANUpSampleBlock(
                in_channels=in_chans, out_channels=out_chans,
                conv_op=conv_op, interpolate_mode=self.interpolate, stride=strides[i],
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

class KM_UNet_3D(nn.Module):
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
                 deep_supervision: bool = False,
                 nonlin_first: bool = False,
                 kan_layer_num: int = 2,
                 spatial_dim: int = 3,
                 img_size=224,
                 patch_size=16,
                 embed_dims=[256, 320, 512],
                 ss2d_dim=None,  # [16, 32, 128, 160]
                 no_kan=False,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1],
                 **invalid_args,
                 ):
        """
        Args:
            kan_layer_num: KAN 的层数。
            spatial_dim: 空间维度。
        """
        super().__init__()

        # 预设的参数。
        self.n_stages = n_stages
        self.conv_layer_num = n_stages - kan_layer_num
        self.kan_layer_num = kan_layer_num
        self.spatial_dim = spatial_dim
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.ss2d_enable = ss2d_dim is not None
        self.ema_enable = False

        if spatial_dim == 2:
            self.pool_op = F.max_pool2d
            self.ss2d_class = SS2D
            self.ema_class = EMA
        elif spatial_dim == 3:
            self.pool_op = F.max_pool3d
        else:
            raise ValueError(f"Unsupported spatial dimension: {spatial_dim}.")

        kan_input_dim = embed_dims[0]

        # 卷积，直到特征图数量达到 kan_input_dim。
        self.encoder_list = nn.ModuleList()
        self.ss2d_encode_list = nn.ModuleList()
        self.ema_encode_list = nn.ModuleList()
        in_chans, out_chans = input_channels, kan_input_dim // (2 ** (self.conv_layer_num - 1))
        conv_dim_list = []
        for i in range(self.conv_layer_num):
            self.encoder_list.append(nn.Sequential(*[BasicConvBlock(
                input_channels=in_chans, output_channels=out_chans,
                conv_op=conv_op, kernel_size=3, stride=1,
                norm_op=norm_op, norm_op_kwargs=norm_op_kwargs
            ), BasicConvBlock(
                input_channels=out_chans, output_channels=out_chans,
                conv_op=conv_op, kernel_size=3, stride=1,
                norm_op=norm_op, norm_op_kwargs=norm_op_kwargs
            )]))
            if self.ss2d_enable:
                self.ss2d_encode_list.append(self.ss2d_class(d_model=ss2d_dim[i]))
            if self.ema_enable:
                self.ema_encode_list.append(self.ema_class(channels=ss2d_dim[i]))

            in_chans = out_chans
            out_chans *= 2
            if i == self.conv_layer_num - 2:
                out_chans = kan_input_dim
            else:
                conv_dim_list.append(in_chans)

        conv_dim_list.append(embed_dims)

        # KAN 模块。
        self.patch_embed_list = nn.ModuleList()
        self.kan_encode_blocks = nn.ModuleList()
        self.kan_decode_blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for i in range(self.kan_layer_num):
            self.patch_embed_list.append(PatchEmbed(
                input_channels=embed_dims[i], output_channels=embed_dims[i + 1],
                conv_op=conv_op, patch_size=3, stride=strides[i + self.conv_layer_num],
            ))
            self.kan_encode_blocks.append(nn.Sequential(*[KANBlock(
                in_features=embed_dims[i + 1], conv_op=conv_op,
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
            )]))
            self.kan_decode_blocks.append(nn.Sequential(*[KANBlock(
                dim=embed_dims[self.kan_layer_num - 1 - i],
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer
            )]))
        # self.norm3 = norm_layer(embed_dims[1])
        # self.norm4 = norm_layer(embed_dims[2])
        #
        # self.dnorm3 = norm_layer(embed_dims[1])
        # self.dnorm4 = norm_layer(embed_dims[0])

        self.conv_decode_list = nn.ModuleList()

        self.ss2d_decode_list = nn.ModuleList()
        self.ema_decode_list = nn.ModuleList()
        for i in range(n_stages):
            if i == 0:
                in_chans, out_chans = embed_dims[0], embed_dims[0]
            else:
                in_chans, out_chans = conv_dim_list[i - 1], conv_dim_list[i]
            self.conv_decode_list.append(nn.Sequential(*[BasicConvBlock(
                input_channels=in_chans,
                output_channels=in_chans,
                conv_op=conv_op,
                kernel_size=3,
                stride=1,
                norm_op=norm_op, norm_op_kwargs=norm_op_kwargs
            ), BasicConvBlock(
                input_channels=in_chans,
                output_channels=out_chans,
                conv_op=conv_op,
                kernel_size=3,
                stride=1,
                norm_op=norm_op, norm_op_kwargs=norm_op_kwargs
            )]))
            if i < n_stages - 1:
                if self.ss2d_enable:
                    self.ss2d_decode_list.append(self.ss2d_class(d_model=ss2d_dim[i]))
                if self.ema_enable:
                    self.ema_decode_list.append(self.ema_class(channels=ss2d_dim[i]))

        # self.block1 = nn.ModuleList([KANBlock(
        #     dim=embed_dims[1],
        #     drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
        # )])
        #
        # self.block2 = nn.ModuleList([KANBlock(
        #     dim=embed_dims[2],
        #     drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
        # )])

        # self.dblock1 = nn.ModuleList([KANBlock(
        #     dim=embed_dims[1],
        #     drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
        # )])
        #
        # self.dblock2 = nn.ModuleList([KANBlock(
        #     dim=embed_dims[0],
        #     drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
        # )])

        # self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
        #                                embed_dim=embed_dims[1])
        # self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
        #                                embed_dim=embed_dims[2])
        #
        # self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])
        # self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])
        # self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0] // 4)
        # self.decoder4 = D_ConvLayer(embed_dims[0] // 4, embed_dims[0] // 8)
        # self.decoder5 = D_ConvLayer(embed_dims[0] // 8, embed_dims[0] // 8)

        self.final = nn.Conv2d(conv_dim_list[0], num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim=1)
        # self.cbam = CBAM(channel=16)
        # self.cbam1 = CBAM(channel=32)
        # self.cbam2 = CBAM(channel=128)
        # # SS2D模块
        # self.ss2d_1 = SS2D(d_model=16)  # Stage 1后
        # self.ss2d_2 = SS2D(d_model=32)  # Stage 2后
        # self.ss2d_3 = SS2D(d_model=128)  # Stage 3后
        # self.ss2d_decoder1 = SS2D(d_model=160)  # Decoder1后
        # self.ss2d_decoder2 = SS2D(d_model=128)  # Decoder2后
        # self.ss2d_decoder3 = SS2D(d_model=32)  # Decoder3后
        # self.ss2d_decoder4 = SS2D(d_model=16)  # Decoder4后
        # # EMA注意力机制
        # self.ema1 = EMA(channels=16)
        # self.ema2 = EMA(channels=32)
        # self.ema3 = EMA(channels=128)
        # self.ema_decoder1 = EMA(channels=160)
        # self.ema_decoder2 = EMA(channels=128)
        # self.ema_decoder3 = EMA(channels=32)
        # self.ema_decoder4 = EMA(channels=16)

    def forward(self, x):
        skips = []
        B = x.size(0)
        for i in range(self.conv_layer_num):
            x = F.relu(self.pool_op(self.encoder_list[i](x), self.kernel_sizes[i], self.strides[i]))
            t = x  # B C H W D
            # t = self.cbam(t)
            if self.ss2d_enable:
                t = helper.channel_to_the_last(t)
                t = self.ss2d_encode_list[i](t)  # SS2D的d_model设置为16
                t = helper.channel_to_the_second(t)
            if self.ema_enable:
                t = self.ema_encode_list[i](t)  # Apply EMA
            skips.append(t)

        for i in range(self.kan_layer_num):
            x, patch = self.patch_embed_list[i](x)  # B C H W D -> B, N, C
            x = self.kan_encode_blocks[i](x, patch) # B, N, C
            x = x.reshape(B, *patch, -1)  # B, N, C -> B, H, W, D, C
            x = helper.channel_to_the_second(x)  # B, C, H, W, D
            t = x
            # 忽略最后一层。
            if i < self.kan_layer_num - 1:
                skips.append(t)

        for i in range(1, self.n_stages):
            x = F.relu(F.interpolate(self.conv_decode_list[-i](x), scale_factor=(2, 2), mode='bilinear'))
            x = torch.add(x, skips[-i])
            if self.ss2d_enable:
                x = helper.channel_to_the_last(x)
                x = self.ss2d_decode_list[-i](x)
                x = helper.channel_to_the_second(x)
            if self.ema_enable:
                x = self.ema_decode_list[-i](x)

        x = F.relu(F.interpolate(self.conv_decode_list[0](x), scale_factor=(2, 2), mode='bilinear'))

        return self.final(x)
