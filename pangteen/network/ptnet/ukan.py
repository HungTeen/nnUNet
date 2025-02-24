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


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 conv_op: Type[_ConvNd],
                 patch_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 ):
        super().__init__()
        self.proj = BasicConvBlock(
            input_channels=input_channels,
            output_channels=output_channels,
            conv_op=conv_op,
            kernel_size=patch_size,
            stride=stride,
        )
        self.norm = nn.LayerNorm(output_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        size = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        x = self.norm(x)

        return x, size


class KANBlock(nn.Module):
    def __init__(self,
                 in_features,
                 conv_op: Type[_ConvNd],
                 hidden_features=None,
                 out_features=None,
                 norm_layer=nn.LayerNorm,
                 drop=0.,
                 drop_path=0.,
                 no_kan=False,
                 layer_count=3,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 grid_size=5,
                 spline_order=3,
                 scale_noise=0.1,
                 scale_base=1.0,
                 scale_spline=1.0,
                 base_activation=torch.nn.SiLU,
                 grid_eps=0.02,
                 grid_range=[-1, 1],
        ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.layers = nn.ModuleList()
        if not no_kan:
            self.layers.append(KANLinear(
                in_features, hidden_features,
                grid_size=grid_size, spline_order=spline_order,
                scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline,
                base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range,
            ))
            for _ in range(layer_count - 1):
                self.layers.append(KANLinear(
                    hidden_features, out_features,
                    grid_size=grid_size, spline_order=spline_order,
                    scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline,
                    base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range,
                ))
        else:
            self.layers.append(nn.Linear(in_features, hidden_features))
            for _ in range(layer_count - 1):
                self.layers.append(nn.Linear(hidden_features, out_features))

        self.conv_blocks = nn.ModuleList([BasicConvBlock(
            input_channels=hidden_features, output_channels=hidden_features,
            conv_op=conv_op, kernel_size=3, stride=1, conv_bias=True, conv_group=hidden_features,
            norm_op = norm_op, norm_op_kwargs = norm_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
        ) for _ in range(layer_count)])

        self.first_norm = norm_layer(in_features)
        # self.drop = nn.Dropout(drop)
        self.last_drop = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, patch):
        B, N, C = x.shape
        residual = x
        x = self.first_norm(x)  # B, N, C
        for layer, conv in zip(self.layers, self.conv_blocks):
            x = layer(x.reshape(B * N, C))
            x = x.reshape(B, N, C).contiguous()
            x = x.reshape(B, *patch, C)  # B, N, C -> B, H, W, D, C
            x = helper.channel_to_the_second(x)  # B, C, H, W, D
            x = conv(x)
            x = x.flatten(2).transpose(1, 2)  # B, C, H, W, D -> B, N, C

        x = self.last_drop(x)

        return x + residual


class UKAN_3D(nn.Module):
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
                 embed_dims=[256, 320, 512],
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

        if spatial_dim == 2:
            self.pool_op = F.max_pool2d
            self.interpole_mode = 'bilinear'
        elif spatial_dim == 3:
            self.pool_op = F.max_pool3d
            self.interpole_mode = 'trilinear'
        else:
            raise ValueError(f"Unsupported spatial dimension: {spatial_dim}.")

        kan_input_dim = embed_dims[0]

        # 卷积，直到特征图数量达到 kan_input_dim。
        self.encoder_list = nn.ModuleList()
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

            in_chans = out_chans
            out_chans *= 2
            if i == self.conv_layer_num - 1:
                out_chans = kan_input_dim
            else:
                conv_dim_list.append(in_chans)

        conv_dim_list += embed_dims

        # KAN 模块。
        self.patch_embed_list = nn.ModuleList()
        self.kan_encode_blocks = nn.ModuleList()
        self.encode_norm_list = nn.ModuleList()
        self.kan_decode_blocks = nn.ModuleList()
        self.decode_norm_list = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for i in range(self.kan_layer_num):
            self.patch_embed_list.append(PatchEmbed(
                input_channels=embed_dims[i], output_channels=embed_dims[i + 1],
                conv_op=conv_op, patch_size=3, stride=strides[i + self.conv_layer_num],
            ))
            self.kan_encode_blocks.append(nn.ModuleList([KANBlock(
                in_features=embed_dims[i + 1], conv_op=conv_op,
                drop=drop_rate, drop_path=dpr[i], no_kan=no_kan,
                norm_layer=norm_layer, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
            )]))
            self.encode_norm_list.append(norm_layer(embed_dims[i + 1]))
            self.kan_decode_blocks.append(nn.ModuleList([KANBlock(
                in_features=embed_dims[i], conv_op=conv_op,
                drop=drop_rate, drop_path=dpr[self.kan_layer_num - 1 - i], norm_layer=norm_layer,
                norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
            )]))
            self.decode_norm_list.append(norm_layer(embed_dims[i]))

        self.conv_decode_list = nn.ModuleList()
        for i in range(n_stages):
            if i == 0:
                in_chans, out_chans = conv_dim_list[0], conv_dim_list[0]
            else:
                in_chans, out_chans = conv_dim_list[i], conv_dim_list[i - 1]
            self.conv_decode_list.append(nn.Sequential(*[BasicConvBlock(
                input_channels=in_chans, output_channels=in_chans,
                conv_op=conv_op, kernel_size=3, stride=1,
                norm_op=norm_op, norm_op_kwargs=norm_op_kwargs
            ), BasicConvBlock(
                input_channels=in_chans,
                output_channels=out_chans,
                conv_op=conv_op, kernel_size=3, stride=1,
                norm_op=norm_op, norm_op_kwargs=norm_op_kwargs
            )]))

        self.final = BasicConvBlock(
            input_channels=conv_dim_list[0], output_channels=num_classes,
            conv_op=conv_op, kernel_size=1, stride=1,
        )

    def forward(self, x):
        skips = []  #  C1, C2, C3, D1
        B = x.size(0)
        for i in range(self.conv_layer_num):
            padding : Union[list, int] = 0
            if isinstance(self.kernel_sizes[i], int):
                padding = (self.kernel_sizes[i] - 1) // 2
            elif isinstance(self.kernel_sizes[i], list):
                padding = [(a - 1) // 2 for a in self.kernel_sizes[i]]
            x = F.relu(self.pool_op(self.encoder_list[i](x), self.kernel_sizes[i], self.strides[i], padding=padding))
            t = x  # B C H W D
            if len(skips) < self.n_stages - 1:
                skips.append(t)

        for i in range(self.kan_layer_num):
            x, patch = self.patch_embed_list[i](x)  # B C H W D -> B, N, C
            for blk in self.kan_encode_blocks[i]:
                x = blk(x, patch) # B, N, C
            x = self.encode_norm_list[i](x)
            x = x.reshape(B, *patch, -1)  # B, N, C -> B, H, W, D, C
            x = helper.channel_to_the_second(x)  # B, C, H, W, D
            t = x
            # 忽略最后一层。
            if len(skips) < self.n_stages - 1:
                skips.append(t)

        for skip in skips:
            print('skip:', skip.size())

        for i in range(1, self.kan_layer_num + 1):
            x = F.relu(F.interpolate(self.conv_decode_list[-i](x), scale_factor=self.strides[-i], mode=self.interpole_mode))
            x = torch.add(x, skips[-i])
            patch = x.shape[2:]
            x = x.flatten(2).transpose(1, 2)  # B C H W D -> B, N, C
            for blk in self.kan_decode_blocks[-i]:
                x = blk(x, patch)  # B, N, C
            x = self.decode_norm_list[-i](x)
            x = x.reshape(B, *patch, -1)  # B, N, C -> B, H, W, D, C
            x = helper.channel_to_the_second(x)  # B, C, H, W, D

        for i in range(self.kan_layer_num + 1, self.n_stages):
            x = F.relu(F.interpolate(self.conv_decode_list[-i](x), scale_factor=self.strides[-i], mode=self.interpole_mode))
            x = torch.add(x, skips[-i])

        x = F.relu(F.interpolate(self.conv_decode_list[0](x), scale_factor=self.strides[0], mode=self.interpole_mode))

        return self.final(x)


if __name__ == "__main__":
    network = UKAN_3D(
        input_channels=1,
        n_stages=4,
        conv_op=nn.Conv3d,
        kernel_sizes=[3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2],
        num_classes=2,
        norm_op=helper.get_matching_instancenorm(dimension=3),
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'negative_slope': 1e-2, 'inplace': True},
        kan_layer_num=2
    ).cuda()

    x = torch.zeros((2, 1, 64, 128, 128), requires_grad=True).cuda()
    y = torch.rand((2, 2, 64, 128, 128), requires_grad=False).cuda()

    with torch.autocast(device_type='cuda', enabled=True):
        pred = network(x)
        print(pred.size())

        loss = F.cross_entropy(pred, y.argmax(1))
        loss.backward()

        print(loss)

        # 检查梯度
        for name, param in network.named_parameters():
            if param.grad is None:
                print(f"No gradient for {name}")
            else:
                print(f"Gradient for {name}: {param.grad.abs().mean()}")


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(network))