from typing import Union, List, Tuple

import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from torch import nn

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from pangteen.loss.uumamba import AutoWeighted_DC_and_CE_and_Focal_loss
from pangteen.network.ptnet.mamba_unet import MambaUNet
from pangteen.network.ptnet.nnunet import nnUNet
from pangteen.network.ptnet.ptunet import PangTeenUNet
from pangteen.network.ptnet.sfunet import SFUNet
from pangteen.trainer.trainers import HTTrainer

class SFUNetTrainer(HTTrainer):

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

        network = SFUNet(**architecture_kwargs)

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class MambaUNetTrainer(HTTrainer):

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
                                                            n_stages=4,
                                                            print_args=True)

        network = MambaUNet(**architecture_kwargs)

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network

class MambaV2UNetTrainer(HTTrainer):

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

        network = MambaUNet(use_v2=False, **architecture_kwargs)

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network