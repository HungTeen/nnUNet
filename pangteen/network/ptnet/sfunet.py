import os

from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from pangteen.network.common.helper import *
import torch
from torch import nn
from typing import Union, Type, List, Tuple, Optional

from pangteen.network.ptnet.conv_blocks import DownSampleBlock, UpSampleBlock, BasicConvBlock, MultiBasicConvBlock
from pangteen.network.ptnet.fushion_blocks import SelectiveFusionBlock
from pangteen.network.ptnet.nnunet import nnUNet
from pangteen.network.ptnet.ptnet import PangTeenNet
from pangteen.network.ptnet.ptunet import PangTeenUNet


class SFUNet(nnUNet):
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
                 deep_supervision: bool = False,
                 pool: str = 'conv',
                 **invalid_args
                 ):
        super().__init__(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_conv_per_stage,
                            num_classes, n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                            dropout_op_kwargs, nonlin, nonlin_kwargs, enable_skip_layer, None, deep_supervision,
                            pool)
        for s in range(n_stages - 1):
            self.decoder_layers[s] = MultiBasicConvBlock(
                n_conv_per_stage_decoder[s], features_per_stage[s], features_per_stage[s], conv_op,
                kernel_sizes[s], 1, 1, conv_bias,
                norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
            )
            self.skip_merge_blocks.append(SelectiveFusionBlock(features_per_stage[s], conv_op, 'mlp'))
            input_channels = features_per_stage[s]



if __name__ == "__main__":
    network = SFUNet(
        input_channels=1,
        n_stages=6,
        features_per_stage= [32, 64, 128, 256, 320, 320],
        # features_per_stage= [16, 32, 64, 128, 256, 512],
        conv_op= nn.Conv3d,
        kernel_sizes = [[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        strides = [[1, 1, 1], [1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
        n_conv_per_stage=[2, 2, 2, 2, 2, 2],
        num_classes=2,
        n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
        conv_bias= True,
        deep_supervision=True
    ).cuda()

    x = torch.zeros((2, 1, 20, 320, 256), requires_grad=False).cuda()

    with torch.autocast(device_type='cuda', enabled=True):
        print(x.device)
        pred = network(x)
        for y in pred:
            print(y.size())

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(network))