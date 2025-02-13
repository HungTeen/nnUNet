import pydoc
from typing import Union, List, Tuple

import torch
from monai.networks.nets import SegResNet
from torch import nn

from pangteen import config
from pangteen.network.lightm_unet.lightm_unet import LightMUNet
from pangteen.network.ptnet.ptresunet import PangTeenResUNet, PangTeenMultiResUNet
from pangteen.network.segmamba.segmamba import SegMamba
from pangteen.network.unetr.unetr import UNETR
from pangteen.trainer.trainers import HTTrainer


class PTResUNetTrainer(HTTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        architecture_kwargs = HTTrainer.update_network_args(arch_init_kwargs, arch_init_kwargs_req_import,
                                                            num_input_channels, num_output_channels,
                                                            enable_deep_supervision,
                                                            print_args=True)

        network = PangTeenResUNet(**architecture_kwargs)

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class PTMultiResUNetTrainer(HTTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        architecture_kwargs = HTTrainer.update_network_args(arch_init_kwargs, arch_init_kwargs_req_import,
                                                            num_input_channels, num_output_channels,
                                                            enable_deep_supervision,
                                                            print_args=True)

        network = PangTeenMultiResUNet(**architecture_kwargs)

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network