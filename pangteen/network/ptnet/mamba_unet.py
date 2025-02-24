from typing import Union, Type, List, Tuple

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from pangteen.network import cfg
from pangteen.network.network_analyzer import NetworkAnalyzer
from pangteen.network.ptnet.conv_blocks import MultiBasicConvBlock
from pangteen.network.ptnet.mamba_blocks import MambaLayer
from pangteen.network.ptnet.nnunet import nnUNet


class MambaUNet(nnUNet):
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
                 use_v2: bool = False,
                 **invalid_args
                 ):
        super().__init__(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_conv_per_stage,
                            num_classes, n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                            dropout_op_kwargs, nonlin, nonlin_kwargs, enable_skip_layer, 'concat', deep_supervision,
                            pool)
        self.encoder_layers = nn.ModuleList()
        for s in range(n_stages):
            self.encoder_layers.append(nn.Sequential(*[MultiBasicConvBlock(
                n_conv_per_stage[s], input_channels, features_per_stage[s],
                conv_op, kernel_sizes[s], 1, 1, conv_bias,
                norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
            ), MambaLayer(
                features_per_stage[s], use_v2=use_v2
            )
            ]))
            input_channels = features_per_stage[s]



if __name__ == "__main__":
    network = MambaUNet(
        use_v2=False,
        **cfg.stage4_network_args
    ).cuda()

    NetworkAnalyzer(network, print_flops=True).analyze()
