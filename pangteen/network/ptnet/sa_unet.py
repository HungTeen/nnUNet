from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from pangteen.network.common.helper import *
import torch
from torch import nn
from typing import Union, Type, List, Tuple, Optional

from pangteen.network.ptnet.attention_blocks import SpatialAttentionBlock
from pangteen.network.ptnet.conv_blocks import DownSampleBlock, UpSampleBlock, BasicConvBlock, MultiBasicConvBlock
from pangteen.network.ptnet.ptnet import PangTeenNet

class SpatialAttentionUNet(PangTeenNet):
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
        assert len(kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                             "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
        for s in range(n_stages - 1):
            self.down_sample_blocks.append(DownSampleBlock(
                conv_op, features_per_stage[s], features_per_stage[s], kernel_sizes[s], strides[s],
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, pool
            ))
            if s < 3:
                self.encoder_layers.append(nn.Sequential(*[MultiBasicConvBlock(
                    n_conv_per_stage[s] - 1, input_channels, features_per_stage[s],
                    conv_op, kernel_sizes[s], 1, 1, conv_bias,
                    norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
                ), SpatialAttentionBlock(
                    features_per_stage[s], conv_op, 3, [5, 7, 11]
                )]))
            else:
                self.encoder_layers.append(MultiBasicConvBlock(
                    n_conv_per_stage[s] - 1, input_channels, features_per_stage[s],
                    conv_op, kernel_sizes[s], 1, 1, conv_bias,
                    norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
                ))
            self.up_sample_blocks.append(UpSampleBlock(
                conv_op, features_per_stage[s + 1], features_per_stage[s], strides[s + 1], conv_bias
            ))
            self.decoder_layers.append(MultiBasicConvBlock(
                n_conv_per_stage_decoder[s], features_per_stage[s] * 2, features_per_stage[s], conv_op,
                kernel_sizes[s], 1, 1, conv_bias,
                norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
            ))
            self.seg_layers.append(BasicConvBlock(
                features_per_stage[s], num_classes, conv_op, 1, 1, 1, conv_bias
            ))
            input_channels = features_per_stage[s]

        self.bottle_neck = MultiBasicConvBlock(
            n_conv_per_stage[-1], features_per_stage[-2], features_per_stage[-1],
            conv_op, kernel_sizes[-1], strides[-1], 1, conv_bias,
            norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
        )


if __name__ == "__main__":
    network = SpatialAttentionUNet(
        input_channels=1,
        n_stages=6,
        features_per_stage= [32, 64, 128, 256, 320, 320],
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