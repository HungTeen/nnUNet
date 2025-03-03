from typing import Type, Union, List

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd

from pangteen.network.common import helper
from pangteen.network.common.kan import KANLinear
from pangteen.network.ptnet.conv_blocks import BasicConvBlock
from pangteen.network.ptnet.mamba_blocks import MambaLayer
from pangteen.network.ptnet.ukan import KANBlock


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

class MultiScaleFusionBlock(nn.Module):

    def __init__(self, features: list, conv_op: Type[_ConvNd],
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 norm_layer=nn.LayerNorm,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 no_kan=False,
                 use_mamba_v2=False):
        super().__init__()
        self.features = features
        feature_sum = sum(features)
        mid = (len(features) - 1) / 2.
        dis = len(features) // 2
        self.samples = nn.ModuleList()
        for i in range(len(features)):
            if i < mid:
                stride = 2**(dis-i)
                self.samples.append(nn.AvgPool3d(kernel_size=stride, stride=stride))
            elif i > mid:
                scale = 2**(i-dis)
                self.samples.append(nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=True))
            else:
                self.samples.append(None)
        self.mamba = MambaLayer(dim=feature_sum, channel_token=False, use_v2=use_mamba_v2)
        self.kan_block = KANBlock(feature_sum, conv_op=conv_op,
            drop=drop_rate, drop_path=drop_path_rate, no_kan=no_kan,
            norm_layer=norm_layer, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, layer_count=1)

    def forward(self, skips):
        size_list = []
        for i, skip in enumerate(skips):
            if self.samples[i] is not None:
                skip = self.samples[i](skip)
            skip, size = helper.to_token(skip)
            size_list.append(size)
            skips[i] = skip

        x = torch.cat(skips, dim=1)
        x = self.mamba(x)
        # x = x.permute(0, 2, 1)
        # x = self.kan_block(x, size_list[2])
        # x = x.permute(0, 2, 1)
        # 将 x 还原为之前的列表。
        feature_sum = 0
        for i, feature in enumerate(self.features):
            skip = x[:, feature_sum:feature_sum+feature]
            skip = helper.to_patch(skip, size_list[i])
            if self.samples[-i-1] is not None:
                skip = self.samples[-i-1](skip)
            feature_sum += feature
            skips[i] = skip
        return skips

if __name__ == '__main__':
    block = MultiScaleFusionBlock([32, 32, 32, 32, 32], nn.Conv3d).cuda()
    x0 = torch.randn(2, 32, 32, 32, 32).cuda()
    x1 = torch.randn(2, 32, 16, 16, 16).cuda()
    x2 = torch.randn(2, 32, 8, 8, 8).cuda()
    x3 = torch.randn(2, 32, 4, 4, 4).cuda()
    x4 = torch.randn(2, 32, 2, 2, 2).cuda()
    skips = [x0, x1, x2, x3, x4]
    block(skips)
    for i, skip in enumerate(skips):
        print(f"Skip {i}: {skip.size()}")