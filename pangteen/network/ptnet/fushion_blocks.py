from typing import Type, Union, List

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd

from pangteen.network.common.kan import KANLinear
from pangteen.network.ptnet.conv_blocks import BasicConvBlock

class SelectiveFusionBlock(nn.Module):
    """
    实现 SIB-UNet 和 SMM-UNet 中的Selective Fusion Block。
    """

    def __init__(self,
                 channels : int,
                 conv_op: Type[_ConvNd],
                 proj_type: str = 'mlp', # 选择投影层的类型，可以是'conv'或者'mlp'或者'kan'.
        ):
        super().__init__()
        # 投影层，特征通道数缩小为原来的1/2。
        self.projection = None
        if proj_type == 'mlp':
            self.projection = nn.Linear(channels * 2, channels)
        elif proj_type == 'kan':
            self.projection = KANLinear(channels * 2, channels)
        elif proj_type == 'conv':
            self.projection = BasicConvBlock(channels * 2, channels, conv_op, 1, 1)
        else:
            raise ValueError(f"Unknown projection type: {proj_type}")
        self.nonlin = nn.Softmax(dim=1)


    def forward(self, x1, x2):
        """
        Args:
            x1: 输入1, shape: (B, C, H, W, D)。
            x2: 输入2, shape: (B, C, H, W, D)。
        """
        # 连接两个输入
        x = torch.cat([x1, x2], dim=1)  # shape: (B, 2C, H, W, D)
        # 全局平均池化
        x = torch.mean(x, dim=[2, 3, 4], keepdim=True) # shape: (B, 2C, 1, 1, 1)
        # 投影
        # 将channel变为最后一个维度。
        x = x.permute(0, 2, 3, 4, 1)  # shape: (B, 1, 1, 1, 2C)
        x = self.projection(x)  # shape: (B, 1, 1, 1, C)
        x = x.permute(0, 4, 1, 2, 3)  # shape: (B, C, 1, 1, 1)
        # softmax
        x = self.nonlin(x)  # shape: (B, C, 1, 1, 1)
        y = 1 - x
        x1 = x1 * x  # shape: (B, C, H, W, D)
        x2 = x2 * y  # shape: (B, C, H, W, D)
        return x1 + x2