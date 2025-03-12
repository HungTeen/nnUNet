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

class ShiftedBlock(nn.Module):

    def __init__(self, dim, shift_size=5):
        super(ShiftedBlock, self).__init__()
        self.dim = dim
        self.shift_size = shift_size
        self.pad = shift_size // 2

    def forward(self, x, patch):
        B, N, C = x.shape
        xn = x.transpose(1, 2).view(B, C, *patch).contiguous()
        xn = F.pad(xn, [self.pad] * 2 * len(patch), "constant", 0)
        # 按通道拆分成若干块。
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, self.dim) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        for i in range(len(patch)):
            x_cat = torch.narrow(x_cat, i + 2, self.pad, patch[i])
        x = x_cat.flatten(2).transpose(1, 2)  # B, C, H, W, D -> B, N, C
        return x

class ShiftedKANBlock(nn.Module):
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
                 shift_size=5,
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

        self.shifted_blocks = nn.ModuleList([ShiftedBlock(
            dim=i%3+2, shift_size=shift_size
        ) for i in range(layer_count)])

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
        for i, layer in enumerate(self.layers):
            x = layer(x.reshape(B * N, C))
            x = self.shifted_blocks[i](x, patch)
            x = x.reshape(B, N, C).contiguous()
            x = x.reshape(B, *patch, C)  # B, N, C -> B, H, W, D, C
            x = helper.channel_to_the_second(x)  # B, C, H, W, D
            x = self.conv_blocks[i](x)
            x = x.flatten(2).transpose(1, 2)  # B, C, H, W, D -> B, N, C

        x = self.last_drop(x)

        return x + residual

if __name__ == '__main__':
    size = 3
    shift_size = 2
    pad = 1
    x = torch.arange(0, (size**3)*2).reshape(1, size**3, 2)
    print(x.transpose(1, 2).view(x.shape[0], x.shape[2], size, size, size).contiguous())
    block = ShiftedBlock(2, shift_size=shift_size)
    x = block(x, [size, size, size])
    print(x.transpose(1, 2).view(x.shape[0], x.shape[2], size, size, size).contiguous())
    block = ShiftedBlock(3, shift_size=shift_size)
    x = block(x, [size, size, size])
    print(x.transpose(1, 2).view(x.shape[0], x.shape[2], size, size, size).contiguous())
    block = ShiftedBlock(4, shift_size=shift_size)
    x = block(x, [size, size, size])
    print(x.transpose(1, 2).view(x.shape[0], x.shape[2], size, size, size).contiguous())