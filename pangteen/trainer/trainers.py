import pydoc
from typing import Union, List, Tuple

import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from pangteen.network.unet.unet import PlainConvUNet


class HTTrainer(nnUNetTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500


    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        让模型选择变成继承制的，不用每次去改 Plan。
        """
        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

        if enable_deep_supervision is not None:
            architecture_kwargs['deep_supervision'] = enable_deep_supervision

        architecture_kwargs['input_channels'] = num_input_channels
        architecture_kwargs['num_classes'] = num_output_channels

        print("Look ! It's UNet : {}", architecture_kwargs)
        network = PlainConvUNet(**architecture_kwargs)

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network