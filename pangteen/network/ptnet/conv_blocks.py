from typing import Type, Union, List, Tuple

import numpy as np
import torch
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

class BasicConvBlock(nn.Module):
    """
    通用的基础卷积块，包含卷积、规范化、dropout和非线性激活函数。
    """
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 conv_op: Type[_ConvNd],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_group: int = 1,
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False
                 ):
        super(BasicConvBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        ops = []

        self.conv = conv_op(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding=[(i - 1) // 2 for i in kernel_size],
            dilation=1,
            bias=conv_bias,
            groups=conv_group,
        )
        ops.append(self.conv)

        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)
            ops.append(self.dropout)

        if norm_op is not None:
            self.norm = norm_op(output_channels, **norm_op_kwargs)
            ops.append(self.norm)

        if nonlin is not None:
            self.nonlin = nonlin(**nonlin_kwargs)
            ops.append(self.nonlin)

        if nonlin_first and (norm_op is not None and nonlin is not None):
            ops[-1], ops[-2] = ops[-2], ops[-1]

        self.all_modules = nn.Sequential(*ops)

    def forward(self, x):
        return self.all_modules(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output_size = [i // j for i, j in zip(input_size, self.stride)]  # we always do same padding
        return np.prod([self.output_channels, *output_size], dtype=np.int64)


class ChannelAttentionBlock(nn.Module):
    """
    MedNeXt中的通道注意力块。
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_op: Type[_ConvNd],
                 do_res: int = True,
                 exp_r: int = 4,
                 kernel_size: int = 7,
                 n_groups: int or None = None,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 grn=False
                 ):

        super().__init__()

        self.do_res = do_res

        # DepthWise Convolution & Normalization
        self.conv1 = BasicConvBlock(
            in_channels,
            in_channels,
            conv_op=conv_op,
            kernel_size=kernel_size,
            stride=1,
            conv_group=in_channels if n_groups is None else n_groups,
            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs
        )

        # Expansion Convolution with Conv3D 1x1x1
        self.expand_conv = BasicConvBlock(
            in_channels,
            exp_r * in_channels,
            conv_op=conv_op,
            kernel_size=1,
            stride=1,
            nonlin=nn.GELU
        )

        # Compression Convolution with Conv3D 1x1x1
        self.compress_conv = BasicConvBlock(
            exp_r * in_channels,
            out_channels,
            conv_op=conv_op,
            kernel_size=1,
            stride=1,
        )

        self.grn = grn
        if grn:
            self.grn_beta = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1, 1), requires_grad=True)
            self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1, 1), requires_grad=True)


    def forward(self, x):
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.expand_conv(x1)
        if self.grn:
            # gamma, beta: learnable affine transform parameters
            # X: input of shape (N,C,H,W,D)
            gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        x1 = self.compress_conv(x1)
        if self.do_res:
            x1 = x + x1
        return x1


class GSC(nn.Module):
    """
    Gated Spatial Convolution (GSC) block：用于在空间信息扁平化之前对特征进行捕获。
    """

    def __init__(self, in_channels,
                 conv_op=nn.Conv3d,
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin_first: bool = False) -> None:
        super().__init__()

        nonlin = nn.ReLU
        self.proj1 = BasicConvBlock(in_channels, in_channels, conv_op=conv_op, kernel_size=3, stride=1,
                                    conv_bias=conv_bias, norm_op=norm_op,
                                    dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs, nonlin=nonlin,
                                    nonlin_first=nonlin_first)

        self.proj2 = BasicConvBlock(in_channels, in_channels, conv_op=conv_op, kernel_size=3, stride=1,
                                    conv_bias=conv_bias, norm_op=norm_op,
                                    dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs, nonlin=nonlin,
                                    nonlin_first=nonlin_first)

        self.proj3 = BasicConvBlock(in_channels, in_channels, conv_op=conv_op, kernel_size=1, stride=1,
                                    conv_bias=conv_bias, norm_op=norm_op,
                                    dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs, nonlin=nonlin,
                                    nonlin_first=nonlin_first)

        self.proj4 = BasicConvBlock(in_channels, in_channels, conv_op=conv_op, kernel_size=1, stride=1,
                                    conv_bias=conv_bias, norm_op=norm_op,
                                    dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs, nonlin=nonlin,
                                    nonlin_first=nonlin_first)



    def forward(self, x):
        x_residual = x

        x1 = self.proj1(x)
        x1 = self.proj2(x1)

        x2 = self.proj3(x)

        x = x1 + x2
        x = self.proj4(x)

        return x + x_residual