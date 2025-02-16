from typing import Type, Union, List, Tuple

import numpy as np
import torch
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from pangteen.network.common.helper import get_matching_pool_op, get_matching_convtransp
from pangteen.network.unet.block import StackedConvBlocks


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


class MultiBasicConvBlock(nn.Module):
    """
    多个基础卷积块，包含卷积、规范化、dropout和非线性激活函数。
    """

    def __init__(self,
                 num_convs: int,
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
                 nonlin_first: bool = False,
                 reverse_order: bool = False
                 ):
        super(MultiBasicConvBlock, self).__init__()
        self.num_convs = num_convs
        if num_convs > 0:
            if not reverse_order:
                self.convs = nn.Sequential(
                    BasicConvBlock(
                        input_channels, output_channels,
                        conv_op, kernel_size, stride, conv_group, conv_bias, norm_op,
                        norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
                    ),
                    *[
                        BasicConvBlock(
                            output_channels, output_channels, conv_op, kernel_size, 1, conv_group, conv_bias, norm_op,
                            norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
                        )
                        for i in range(1, num_convs)
                    ]
                )
            else:
                self.convs = nn.Sequential(
                    *[
                        BasicConvBlock(
                            input_channels, input_channels, conv_op, kernel_size, 1, conv_group, conv_bias, norm_op,
                            norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
                        )
                        for i in range(1, num_convs)
                    ],
                    BasicConvBlock(
                        input_channels, output_channels,
                        conv_op, kernel_size, stride, conv_group, conv_bias, norm_op,
                        norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
                    ),
                )

    def forward(self, x):
        return self.convs(x) if self.num_convs > 0 else x


class DownSampleBlock(nn.Module):
    """
    下采样块，包含卷积、规范化、dropout和非线性激活函数。
    """

    def __init__(self, conv_op: Type[_ConvNd],
                 input_channels: int, out_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]], stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False, norm_op: Union[None, Type[nn.Module]] = None, norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None, dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None, nonlin_kwargs: dict = None,
                 pool: str = 'conv'):
        super().__init__()
        stage_modules = []
        if pool == 'max' or pool == 'avg':
            stage_modules.append(get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=stride, stride=stride))
            conv_stride = 1
        elif pool == 'conv':
            conv_stride = stride
        else:
            raise RuntimeError()
        stage_modules.append(BasicConvBlock(
            input_channels, out_channels, conv_op, kernel_size, conv_stride, 1,
            conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
        ))
        self.encoder = nn.Sequential(*stage_modules)

    def forward(self, x):
        return self.encoder(x)


class UpSampleBlock(nn.Module):
    """
    下采样块。
    """

    def __init__(self, conv_op: Type[_ConvNd],
                 input_channels: int, out_channels: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False):
        super().__init__()
        trans_conv_op = get_matching_convtransp(conv_op)
        self.up_sample = trans_conv_op(
            input_channels, out_channels, stride, stride,
            bias=conv_bias
        )

    def forward(self, x):
        return self.up_sample(x)


class BasicResBlock(nn.Module):
    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            conv_op: Type[_ConvNd],
            kernel_size: Union[int, List[int], Tuple[int, ...]],
            stride: Union[int, List[int], Tuple[int, ...]],
            conv_group: int = 1,
            conv_bias: bool = False,
            norm_op: Union[None, Type[nn.Module]] = None,
            norm_op_kwargs: dict = None,
            nonlin: Union[None, Type[torch.nn.Module]] = None,
            nonlin_kwargs: dict = None,
            use_1x1conv: bool = False
    ):
        super().__init__()
        self.conv1 = BasicConvBlock(
            input_channels, output_channels, conv_op, kernel_size, stride, conv_group, conv_bias, norm_op,
            norm_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs
        )
        self.conv2 = BasicConvBlock(
            output_channels, output_channels, conv_op, kernel_size, 1, conv_group, conv_bias, norm_op,
            norm_op_kwargs
        )
        self.nonlin = nonlin(**nonlin_kwargs)

        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.nonlin(y)


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
