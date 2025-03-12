from typing import Type, Union, List

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd

from pangteen.network.ptnet.conv_blocks import BasicConvBlock


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


class SpatialAttentionBlock(nn.Module):
    """
    对CPCANet中的空间注意力块进行扩展，支持3D模型。
    """

    def __init__(self,
                 channels: int,
                 conv_op: Type[_ConvNd],
                 dim: int = 3,
                 spatial_len: List[int] = [7, 11, 21],
                 channelAttention_reduce=4
                 ):
        super().__init__()

        self.attention_conv = BasicConvBlock(
            channels, channels, conv_op,
            kernel_size=1, stride=1, conv_group=channels,
            nonlin=nn.GELU
        )
        self.conv = BasicConvBlock(
            channels, channels, conv_op,
            kernel_size=5, stride=1, conv_group=channels,
        )
        self.spatial_convs = nn.ModuleList()
        for i, len in enumerate(spatial_len):
            convs = []
            for d in range(dim):
                kernel_size = [1] * dim
                kernel_size[d] = len
                convs.append(BasicConvBlock(
                    channels, channels, conv_op,
                    kernel_size=kernel_size, stride=1, conv_group=channels
                ))
            self.spatial_convs.append(nn.Sequential(*convs))

        # self.ca = ChannelAttention(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)

    def forward(self, x):
        # x = self.first_conv(x)  #   Global Perceptron

        # channel_att_vec = self.ca(inputs)
        # inputs = channel_att_vec * inputs
        spatial_x = self.conv(x)
        for i, conv in enumerate(self.spatial_convs):
            x = x + conv(spatial_x)

        # spatial_att = self.conv(x)

        # out = spatial_att * inputs
        # out = self.conv(out)
        return x


class Spatial_Att_Bridge(nn.Module):
    """
    SAB Block
    """
    def __init__(self, shared_conv2d):
        super().__init__()
        self.shared_conv2d = shared_conv2d

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.cat([avg_out, max_out], dim=1)
        att = self.shared_conv2d(att)
        att *= x
        x = x + att
        return x


class Channel_Att_Bridge(nn.Module):
    """
    CAB Block
    """
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = nn.Linear(c_list_sum, c_list[3]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)
        self.att5 = nn.Linear(c_list_sum, c_list[4]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[4], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3, t4, t5):
        att = torch.cat((self.avgpool(t1),
                         self.avgpool(t2),
                         self.avgpool(t3),
                         self.avgpool(t4),
                         self.avgpool(t5)), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        att5 = self.sigmoid(self.att5(att))
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t4)
            att5 = att5.transpose(-1, -2).unsqueeze(-1).expand_as(t5)
        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).expand_as(t4)
            att5 = att5.unsqueeze(-1).expand_as(t5)

        return att1, att2, att3, att4, att5
