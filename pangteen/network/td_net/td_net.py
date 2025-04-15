import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

import torch
import torch.nn as nn

class ConvBlock(torch.nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            stride=1,
            batch_norm=True,
            preactivation=False,
            dim="3d"
    ):
        super().__init__()

        if dim == "3d":
            constant_pad = torch.nn.ConstantPad3d
            conv = torch.nn.Conv3d
            bn = torch.nn.BatchNorm3d
        elif dim == "2d":
            constant_pad = torch.nn.ConstantPad2d
            conv = torch.nn.Conv2d
            bn = torch.nn.BatchNorm2d
        else:
            raise RuntimeError(f"{dim} dimension is error")

        padding = kernel_size - stride
        if padding % 2 != 0:
            pad = constant_pad(
                tuple([padding % 2, padding - padding % 2] * (3 if dim == "3d" else 2)), 0
            )
        else:
            pad = constant_pad(padding // 2, 0)

        if preactivation:
            layers = [
                torch.nn.ReLU(),
                pad,
                conv(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
            ]
            if batch_norm:
                layers = [bn(in_channel)] + layers
        else:
            layers = [
                pad,
                conv(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
            ]
            if batch_norm:
                layers.append(bn(out_channel))
            layers.append(torch.nn.ReLU())

        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class DepthWiseSeparateConvBlock(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            stride=1,
            batch_norm=True,
            preactivation=False,
            dim="3d"
    ):
        super(DepthWiseSeparateConvBlock, self).__init__()

        if dim == "3d":
            constant_pad = torch.nn.ConstantPad3d
            conv = torch.nn.Conv3d
            bn = torch.nn.BatchNorm3d
        elif dim == "2d":
            constant_pad = torch.nn.ConstantPad2d
            conv = torch.nn.Conv2d
            bn = torch.nn.BatchNorm2d
        else:
            raise RuntimeError(f"{dim} dimension is error")

        padding = kernel_size - stride
        if padding % 2 != 0:
            pad = constant_pad(
                tuple([padding % 2, padding - padding % 2] * (3 if dim == "3d" else 2)), 0
            )
        else:
            pad = constant_pad(padding // 2, 0)

        if preactivation:
            layers = [
                torch.nn.ReLU(),
                pad,
                conv(
                    in_channels=in_channel,
                    out_channels=in_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=in_channel,
                    bias=False
                ),
                conv(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    bias=True
                )
            ]
            if batch_norm:
                layers = [bn(in_channel)] + layers
        else:
            layers = [
                pad,
                conv(
                    in_channels=in_channel,
                    out_channels=in_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=in_channel,
                    bias=False
                ),
                conv(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    bias=False
                )
            ]
            if batch_norm:
                layers.append(bn(out_channel))
            layers.append(torch.nn.ReLU())

        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class DenseFeatureStackWithLocalBlock(nn.Module):

    def __init__(self, in_channel, kernel_size, unit, growth_rate, dim="3d"):
        super(DenseFeatureStackWithLocalBlock, self).__init__()

        self.conv_units = torch.nn.ModuleList()
        for i in range(unit):
            self.conv_units.append(
                ConvBlock(
                    in_channel=in_channel,
                    out_channel=growth_rate,
                    kernel_size=kernel_size,
                    stride=1,
                    batch_norm=True,
                    preactivation=True,
                    dim=dim
                )
            )
            in_channel += growth_rate

    def forward(self, x):
        stack_feature = None

        for i, conv in enumerate(self.conv_units):
            if stack_feature is None:
                inputs = x
            else:
                inputs = torch.cat([x, stack_feature], dim=1)
            out = conv(inputs)
            if stack_feature is None:
                stack_feature = out
            else:
                stack_feature = torch.cat([stack_feature, out], dim=1)

        return torch.cat([x, stack_feature], dim=1)

class DownSampleWithLocalBlock(nn.Module):

    def __init__(self, in_channel, base_channel, kernel_size, unit, growth_rate, skip_channel=None, downsample=True, skip=True, dim="3d"):
        super(DownSampleWithLocalBlock, self).__init__()
        self.skip = skip

        self.downsample = ConvBlock(
            in_channel=in_channel,
            out_channel=base_channel,
            kernel_size=kernel_size,
            stride=(2 if downsample else 1),
            batch_norm=True,
            preactivation=True,
            dim=dim
        )

        self.dfs_with_pmfs = DenseFeatureStackWithLocalBlock(
            in_channel=base_channel,
            kernel_size=3,
            unit=unit,
            growth_rate=growth_rate,
            dim=dim
        )

        if skip:
            self.skip_conv = ConvBlock(
                in_channel=base_channel + unit * growth_rate,
                out_channel=skip_channel,
                kernel_size=3,
                stride=1,
                batch_norm=True,
                preactivation=True,
                dim=dim
            )

    def forward(self, x):
        x = self.downsample(x)
        x = self.dfs_with_pmfs(x)

        if self.skip:
            x_skip = self.skip_conv(x)
            return x, x_skip
        else:
            return x


class GlobalBlock_AP_Separate(nn.Module):
    """
    Global polarized multi-scale feature self-attention module using global multi-scale features
    to expand the number of attention points and thus enhance features at each scale,
    replacing standard convolution with depth-wise separable convolution
    """
    def __init__(self, in_channels, max_pool_kernels, ch, ch_k, ch_v, br, dim="3d"):
        """
        Initialize a global polarized multi-scale feature self-attention module that replaces standard convolution with depth-wise separable convolution

        :param in_channels: channels of each scale feature map
        :param max_pool_kernels: sizes of downsample kernels for feature maps at each scale
        :param ch: channel of global uniform feature
        :param ch_k: channel of K
        :param ch_v: channel of V
        :param br: number of branches
        :param dim: dimension
        """
        super(GlobalBlock_AP_Separate, self).__init__()
        self.ch_bottle = in_channels[-1]
        self.ch = ch
        self.ch_k = ch_k
        self.ch_v = ch_v
        self.br = br
        self.ch_in = self.ch * self.br
        self.dim = dim

        if dim == "3d":
            max_pool = nn.MaxPool3d
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        elif dim == "2d":
            max_pool = nn.MaxPool2d
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        else:
            raise RuntimeError(f"{dim} dimension is error")

        self.ch_convs = nn.ModuleList([
            DepthWiseSeparateConvBlock(
                in_channel=in_channel,
                out_channel=self.ch,
                kernel_size=3,
                stride=1,
                batch_norm=True,
                preactivation=True,
                dim=dim
            )
            for in_channel in in_channels
        ])

        self.max_pool_layers = nn.ModuleList([
            max_pool(kernel_size=k, stride=k)
            for k in max_pool_kernels
        ])

        self.ch_Wq = DepthWiseSeparateConvBlock(in_channel=self.ch_in, out_channel=self.ch_in, kernel_size=1, stride=1, batch_norm=True, preactivation=True, dim=dim)
        self.ch_Wk = DepthWiseSeparateConvBlock(in_channel=self.ch_in, out_channel=1, kernel_size=1, stride=1, batch_norm=True, preactivation=True, dim=dim)
        self.ch_Wv = DepthWiseSeparateConvBlock(in_channel=self.ch_in, out_channel=self.ch_in, kernel_size=1, stride=1, batch_norm=True, preactivation=True, dim=dim)
        self.ch_softmax = nn.Softmax(dim=1)
        self.ch_score_conv = conv(self.ch_in, self.ch_in, 1)
        self.ch_layer_norm = (nn.LayerNorm((self.ch_in, 1, 1, 1)) if dim == "3d" else nn.LayerNorm((self.ch_in, 1, 1)))
        self.sigmoid = nn.Sigmoid()

        self.sp_Wq = DepthWiseSeparateConvBlock(in_channel=self.ch_in, out_channel=self.br * self.ch_k, kernel_size=1, stride=1, batch_norm=True, preactivation=True, dim=dim)
        self.sp_Wk = DepthWiseSeparateConvBlock(in_channel=self.ch_in, out_channel=self.br * self.ch_k, kernel_size=1, stride=1, batch_norm=True, preactivation=True, dim=dim)
        self.sp_Wv = DepthWiseSeparateConvBlock(in_channel=self.ch_in, out_channel=self.br * self.ch_v, kernel_size=1, stride=1, batch_norm=True, preactivation=True, dim=dim)
        self.sp_softmax = nn.Softmax(dim=-1)
        self.sp_output_conv = DepthWiseSeparateConvBlock(in_channel=self.br * self.ch_v, out_channel=self.ch_in, kernel_size=1, stride=1, batch_norm=True, preactivation=True, dim=dim)

        self.output_conv = DepthWiseSeparateConvBlock(in_channel=self.ch_in, out_channel=self.ch_bottle, kernel_size=3, stride=1, batch_norm=True, preactivation=True, dim=dim)

    def forward(self, feature_maps):
        max_pool_maps = [
            max_pool_layer(feature_maps[i])
            for i, max_pool_layer in enumerate(self.max_pool_layers)
        ]
        ch_outs = [
            ch_conv(max_pool_maps[i])
            for i, ch_conv in enumerate(self.ch_convs)
        ]
        x = torch.cat(ch_outs, dim=1)

        if self.dim == "3d":
            bs, c, d, h, w = x.size()

            ch_Q = self.ch_Wq(x)  # bs, self.ch_in, d, h, w
            ch_K = self.ch_Wk(x)  # bs, 1, d, h, w
            ch_V = self.ch_Wv(x)  # bs, self.ch_in, d, h, w
            ch_Q = ch_Q.reshape(bs, -1, d * h * w)  # bs, self.ch_in, d*h*w
            ch_K = ch_K.reshape(bs, -1, 1)  # bs, d*h*w, 1
            ch_K = self.ch_softmax(ch_K)  # bs, d*h*w, 1
            Z = torch.matmul(ch_Q, ch_K).unsqueeze(-1).unsqueeze(-1)  # bs, self.ch_in, 1, 1, 1
            ch_score = self.sigmoid(self.ch_layer_norm(self.ch_score_conv(Z)))  # bs, self.ch_in, 1, 1, 1
            ch_out = ch_V * ch_score  # bs, self.ch_in, d, h, w

            sp_Q = self.sp_Wq(ch_out)  # bs, self.br*self.ch_k, d, h, w
            sp_K = self.sp_Wk(ch_out)  # bs, self.br*self.ch_k, d, h, w
            sp_V = self.sp_Wv(ch_out)  # bs, self.br*self.ch_v, d, h, w
            sp_Q = sp_Q.reshape(bs, self.br, self.ch_k, d, h, w).permute(0, 2, 3, 4, 5, 1).reshape(bs, self.ch_k, -1)  # bs, self.ch_k, d*h*w*self.br
            sp_K = sp_K.reshape(bs, self.br, self.ch_k, d, h, w).permute(0, 2, 3, 4, 5, 1).mean(-1).mean(-1).mean(-1).mean(-1).reshape(bs, 1, self.ch_k)  # bs, 1, self.ch_k
            sp_V = sp_V.reshape(bs, self.br, self.ch_k, d, h, w).permute(0, 2, 3, 4, 5, 1)  # bs, self.ch_v, d, h, w, self.br
            sp_K = self.sp_softmax(sp_K)  # bs, 1, self.ch_k
            Z = torch.matmul(sp_K, sp_Q).reshape(bs, 1, d, h, w, self.br)  # bs, 1, d, h, w, self.br
            sp_score = self.sigmoid(Z)  # bs, 1, d, h, w, self.br
            sp_out = sp_V * sp_score  # bs, self.ch_v, d, h, w, self.br
            sp_out = sp_out.permute(0, 5, 1, 2, 3, 4).reshape(bs, self.br * self.ch_v, d, h, w)  # bs, self.br*self.ch_v, d, h, w
            sp_out = self.sp_output_conv(sp_out)  # bs, self.ch_in, d, h, w

            out = self.output_conv(sp_out)
        else:
            bs, c, h, w = x.size()

            ch_Q = self.ch_Wq(x)  # bs, self.ch_in, h, w
            ch_K = self.ch_Wk(x)  # bs, 1, h, w
            ch_V = self.ch_Wv(x)  # bs, self.ch_in, h, w
            ch_Q = ch_Q.reshape(bs, -1, h * w)  # bs, self.ch_in, h*w
            ch_K = ch_K.reshape(bs, -1, 1)  # bs, h*w, 1
            ch_K = self.ch_softmax(ch_K)  # bs, h*w, 1
            Z = torch.matmul(ch_Q, ch_K).unsqueeze(-1)  # bs, self.ch_in, 1, 1
            ch_score = self.sigmoid(self.ch_layer_norm(self.ch_score_conv(Z)))  # bs, self.ch_in, 1, 1
            ch_out = ch_V * ch_score  # bs, self.ch_in, h, w

            sp_Q = self.sp_Wq(ch_out)  # bs, self.br*self.ch_k, h, w
            sp_K = self.sp_Wk(ch_out)  # bs, self.br*self.ch_k, h, w
            sp_V = self.sp_Wv(ch_out)  # bs, self.br*self.ch_v, h, w
            sp_Q = sp_Q.reshape(bs, self.br, self.ch_k, h, w).permute(0, 2, 3, 4, 1).reshape(bs, self.ch_k, -1)  # bs, self.ch_k, h*w*self.br
            sp_K = sp_K.reshape(bs, self.br, self.ch_k, h, w).permute(0, 2, 3, 4, 1).mean(-1).mean(-1).mean(-1).reshape(bs, 1, self.ch_k)  # bs, 1, self.ch_k
            sp_V = sp_V.reshape(bs, self.br, self.ch_k, h, w).permute(0, 2, 3, 4, 1)  # bs, self.ch_v, h, w, self.br
            sp_K = self.sp_softmax(sp_K)  # bs, 1, self.ch_k
            Z = torch.matmul(sp_K, sp_Q).reshape(bs, 1, h, w, self.br)  # bs, 1, h, w, self.br
            sp_score = self.sigmoid(Z)  # bs, 1, h, w, self.br
            sp_out = sp_V * sp_score  # bs, self.ch_v, h, w, self.br
            sp_out = sp_out.permute(0, 4, 1, 2, 3).reshape(bs, self.br * self.ch_v, h, w)  # bs, self.br*self.ch_v, h, w
            sp_out = self.sp_output_conv(sp_out)  # bs, self.ch_in, h, w

            out = self.output_conv(sp_out)
        return out


class TDNet(nn.Module):
    """
    Reference: https://github.com/yykzjh/PMFSNet
    """
    def __init__(self, in_channels=1, out_channels=35, dim="3d", scaling_version="TINY",
                 basic_module=DownSampleWithLocalBlock,
                 global_module=GlobalBlock_AP_Separate):
        super(TDNet, self).__init__()

        self.scaling_version = scaling_version

        if scaling_version == "BASIC":
            base_channels = [24, 48, 64]
            skip_channels = [24, 48, 64]
            units = [5, 10, 10]
            pmfs_ch = 64
        elif scaling_version == "SMALL":
            base_channels = [24, 24, 24]
            skip_channels = [12, 24, 24]
            units = [5, 10, 10]
            pmfs_ch = 48
        elif scaling_version == "TINY":
            base_channels = [24, 24, 24]
            skip_channels = [12, 24, 24]
            units = [3, 5, 5]
            pmfs_ch = 48
        else:
            raise RuntimeError(f"{scaling_version} scaling version is not available")

        if dim == "3d":
            upsample_mode = 'trilinear'
        elif dim == "2d":
            upsample_mode = 'bilinear'
        else:
            raise RuntimeError(f"{dim} dimension is error")
        kernel_sizes = [5, 3, 3]
        growth_rates = [4, 8, 16]
        downsample_channels = [base_channels[i] + units[i] * growth_rates[i] for i in range(len(base_channels))]

        self.down_convs = nn.ModuleList()
        for i in range(3):
            self.down_convs.append(
                basic_module(
                    in_channel=(in_channels if i == 0 else downsample_channels[i - 1]),
                    base_channel=base_channels[i],
                    kernel_size=kernel_sizes[i],
                    skip_channel=skip_channels[i],
                    unit=units[i],
                    growth_rate=growth_rates[i],
                    downsample=True,
                    skip=((i < 2) if scaling_version == "BASIC" else True),
                    dim=dim
                )
            )

        self.Global = global_module(
            in_channels=downsample_channels,
            max_pool_kernels=[4, 2, 1],
            ch=pmfs_ch,
            ch_k=pmfs_ch,
            ch_v=pmfs_ch,
            br=3,
            dim=dim
        )

        if scaling_version == "BASIC":
            self.up2 = torch.nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.up_conv2 = basic_module(in_channel=downsample_channels[2] + skip_channels[1],
                                         base_channel=base_channels[1],
                                         kernel_size=3,
                                         unit=units[1],
                                         growth_rate=growth_rates[1],
                                         downsample=False,
                                         skip=False,
                                         dim=dim)

            self.up1 = torch.nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.up_conv1 = basic_module(in_channel=downsample_channels[1] + skip_channels[0],
                                         base_channel=base_channels[0],
                                         kernel_size=3,
                                         unit=units[0],
                                         growth_rate=growth_rates[0],
                                         downsample=False,
                                         skip=False,
                                         dim=dim)
        else:
            self.bottle_conv = ConvBlock(
                in_channel=downsample_channels[2] + skip_channels[2],
                out_channel=skip_channels[2],
                kernel_size=3,
                stride=1,
                batch_norm=True,
                preactivation=True,
                dim=dim
            )

            self.upsample_1 = torch.nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.upsample_2 = torch.nn.Upsample(scale_factor=4, mode=upsample_mode)

        self.out_conv = ConvBlock(
            in_channel=(downsample_channels[0] if scaling_version == "BASIC" else sum(skip_channels)),
            out_channel=out_channels,
            kernel_size=3,
            stride=1,
            batch_norm=True,
            preactivation=True,
            dim=dim
        )
        self.upsample_out = torch.nn.Upsample(scale_factor=2, mode=upsample_mode)

    def forward(self, x):
        if self.scaling_version == "BASIC":
            x1, x1_skip = self.down_convs[0](x)
            x2, x2_skip = self.down_convs[1](x1)
            x3 = self.down_convs[2](x2)

            d3 = self.Global([x1, x2, x3])

            d2 = self.up2(d3)
            d2 = torch.cat((x2_skip, d2), dim=1)
            d2 = self.up_conv2(d2)
            d1 = self.up1(d2)
            d1 = torch.cat((x1_skip, d1), dim=1)
            d1 = self.up_conv1(d1)

            out = self.out_conv(d1)
            out = self.upsample_out(out)
        else:
            x1, skip1 = self.down_convs[0](x)
            x2, skip2 = self.down_convs[1](x1)
            x3, skip3 = self.down_convs[2](x2)

            x3 = self.Global([x1, x2, x3])
            skip3 = self.bottle_conv(torch.cat([x3, skip3], dim=1))

            skip2 = self.upsample_1(skip2)
            skip3 = self.upsample_2(skip3)

            out = self.out_conv(torch.cat([skip1, skip2, skip3], dim=1))
            out = self.upsample_out(out)

        return out





if __name__ == '__main__':

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dims = ["3d", "2d"]
    channels = [1, 3]

    scaling_versions = ["BASIC", "SMALL", "TINY"]

    xs = [torch.randn((1, 1, 160, 160, 96)).to(device), torch.randn((1, 3, 224, 224)).to(device)]

    for i, dim in enumerate(dims):
        for scaling_version in scaling_versions:
            model = TDNet(in_channels=channels[i], out_channels=2, dim=dim, scaling_version=scaling_version).to(device)
            y = model(xs[i])
            print(dim + "-" + scaling_version, ":")
            print(xs[i].size())
            print(y.size())
            print("params: {:.6f}M".format(count_parameters(model)))