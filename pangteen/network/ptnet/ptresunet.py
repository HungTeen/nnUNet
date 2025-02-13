import os
import torch.nn.functional as F
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from torch.nn import LeakyReLU
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from pangteen.network.common.helper import *
import torch
from torch import nn
from typing import Union, Type, List, Tuple, Optional

from pangteen.network.ptnet.conv_blocks import DownSampleBlock, UpSampleBlock, BasicConvBlock, MultiBasicConvBlock
from pangteen.network.ptnet.ptnet import PangTeenNet
from pangteen.network.ptnet.ptunet import PangTeenUNet


class SkipResBlock(nn.Module):

    def __init__(self, conv_op: Type[_ConvNd], channels: int, norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None):
        super(SkipResBlock, self).__init__()
        self.conv1 = BasicConvBlock(channels, channels, conv_op, 3, 1, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs)
        self.conv2 = BasicConvBlock(channels, channels, conv_op, 1, 1, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs)
        self.nonlin = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = x1 + x2
        x = self.nonlin(x)
        return x


class MultiResBlock(nn.Module):

    def __init__(self, conv_op: Type[_ConvNd], in_channels: int, out_channels: int,
                 weights: Tuple = (1. / 6, 1. / 3, 1. / 2),
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 ):
        super(MultiResBlock, self).__init__()
        self.conv_blocks = nn.ModuleList()
        input_channels = in_channels
        for idx in range(len(weights)):
            next_channels = int(out_channels * weights[idx])
            BasicConvBlock(input_channels, next_channels, conv_op, 3, 1, 1, conv_bias
                           , norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
                           )
            input_channels = next_channels
        self.conv1 = BasicConvBlock(in_channels, out_channels, conv_op, 1, 1,
                                    1, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin,
                                    nonlin_kwargs)

    def forward(self, x):
        skips = []
        t = x
        for conv_block in self.conv_blocks:
            t = conv_block(t)
            skips.append(t)
        x = self.conv1(x)
        # 将 skips 中的特征图 concat 起来。
        t = torch.cat(skips, dim=1)
        return x + t


class PangTeenResUNet(PangTeenUNet):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 enable_skip_layer: bool = True,
                 skip_merge_type: str = 'concat',
                 deep_supervision: bool = False,
                 pool: str = 'conv',
                 **invalid_args
                 ):
        super().__init__(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_conv_per_stage,
            num_classes, n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs, enable_skip_layer, skip_merge_type, deep_supervision, pool
        )

        for s in range(n_stages - 1):
            self.skip_layers.append(nn.Sequential(*[
                SkipResBlock(conv_op, features_per_stage[s], norm_op, norm_op_kwargs, nonlin, nonlin_kwargs) for _ in range(n_stages - s - 1)
            ]))


class PangTeenMultiResUNet(PangTeenNet):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 enable_skip_layer: bool = True,
                 skip_merge_type: str = 'concat',
                 deep_supervision: bool = False,
                 pool: str = 'conv',
                 **invalid_args
                 ):
        super().__init__(n_stages, enable_skip_layer, skip_merge_type, deep_supervision)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
        for s in range(n_stages - 1):
            self.down_sample_blocks.append(DownSampleBlock(
                conv_op, features_per_stage[s], features_per_stage[s], kernel_sizes[s], strides[s],
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, pool
            ))
            self.encoder_layers.append(MultiResBlock(
                conv_op, input_channels, features_per_stage[s], conv_bias=conv_bias,
                norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
                nonlin=nonlin, nonlin_kwargs=nonlin_kwargs
            ))
            self.up_sample_blocks.append(UpSampleBlock(
                conv_op, features_per_stage[s + 1], features_per_stage[s], strides[s + 1], conv_bias
            ))
            self.decoder_layers.append(MultiResBlock(
                conv_op, features_per_stage[s] * 2, features_per_stage[s], conv_bias=conv_bias,
                norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
                nonlin=nonlin, nonlin_kwargs=nonlin_kwargs
            ))
            self.seg_layers.append(BasicConvBlock(
                features_per_stage[s], num_classes, conv_op, 1, 1, 1, conv_bias
            ))
            self.skip_layers.append(nn.Sequential(*[
                SkipResBlock(conv_op, features_per_stage[s], norm_op, norm_op_kwargs, nonlin, nonlin_kwargs) for _ in
                range(n_stages - s - 1)
            ]))
            input_channels = features_per_stage[s]

        self.bottle_neck = MultiBasicConvBlock(
            n_conv_per_stage[-1], features_per_stage[-2], features_per_stage[-1],
            conv_op, kernel_sizes[-1], strides[-1], 1, conv_bias,
            norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
        )


if __name__ == "__main__":
    # 设置CUDA可见设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    network = PangTeenResUNet(
        input_channels=1,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        # features_per_stage= [16, 32, 64, 128, 256, 512],
        conv_op=nn.Conv3d,
        kernel_sizes=[[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        strides=[[1, 1, 1], [1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
        n_conv_per_stage=[2, 2, 2, 2, 2, 2],
        num_classes=2,
        n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
        conv_bias=True,
        deep_supervision=True
    ).cuda()

    x = torch.zeros((2, 1, 64, 128, 128), requires_grad=True).cuda()
    y = torch.rand((2, 2, 64, 128, 128), requires_grad=False).cuda()

    with torch.autocast(device_type='cuda', enabled=True):
        pred_list = network(x)
        for pred in pred_list:
            y = torch.rand(pred.size(), requires_grad=False).cuda()
            loss = F.cross_entropy(pred, y.argmax(1))
            print(loss)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(count_parameters(network))
