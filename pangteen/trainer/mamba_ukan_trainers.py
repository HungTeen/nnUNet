from typing import Union, List, Tuple

import torch
from torch import nn

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from pangteen.loss.uumamba import AutoWeighted_DC_and_CE_and_Focal_loss, AutoWeighted_DC_and_CE_loss
from pangteen.network.ptnet.mamba_ukan import MambaUKan
from pangteen.network.ptnet.ptukan import UKAN_3D, SFUKAN_3D
from pangteen.network.ukan.ukan_2d import UKAN
from pangteen.trainer.trainers import HTTrainer


class MambaUKanTrainer(HTTrainer):
    """
    A: KAN | B: Mamba | C: SF | D: ResPath | E: Loss | F: DS | G: GSC | H: EPA
    Default is A + B, A + C and A + E in ukan_trainers.py
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500
        self.initial_lr = 1e-3
        self.enable_deep_supervision = False

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
                                                            n_stages=5,
                                                            print_args=True)

        network = MambaUKan(
            encoder_types=['Conv', 'MChannel', 'MChannel', 'KAN', 'KAN'],
            decoder_types=['Conv', 'MChannel', 'MChannel', 'KAN', 'KAN'],
            down_sample_first=True,
            mamba_count=1,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class MoreMambaUKanTrainer(MambaUKanTrainer):
    """
    A: KAN | B: Mamba | C: SF | D: ResPath | E: Loss | F: DS
    Default is A + B, A + C and A + E in ukan_trainers.py
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 300

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
                                                            n_stages=5,
                                                            print_args=True)

        network = MambaUKan(
            encoder_types=['Conv', 'MChannel', 'MChannel', 'KAN', 'KAN'],
            decoder_types=['Conv', 'MChannel', 'MChannel', 'KAN', 'KAN'],
            down_sample_first=True,
            mamba_count=2,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class ResUKanTrainer(MambaUKanTrainer):
    """
    A + D
    """

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
                                                            n_stages=5,
                                                            print_args=True)

        network = MambaUKan(
            res_path_count=[4, 3, 2, 1],
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network

class DSUKanTrainer(MambaUKanTrainer):
    """
    A + F
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = True

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
                                                            n_stages=5,
                                                            print_args=True)

        network = MambaUKan(
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network

class GSCUKanTrainer(MambaUKanTrainer):
    """
    A + G
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = True

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
                                                            n_stages=5,
                                                            print_args=True)

        network = MambaUKan(
            encoder_types=['Conv', 'GSC', 'GSC', 'KAN', 'KAN'],
            decoder_types=['Conv', 'GSC', 'GSC', 'KAN', 'KAN'],
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network

class EPAUKanTrainer(MambaUKanTrainer):
    """
    A + H
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 300

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
                                                            n_stages=5,
                                                            print_args=True)

        network = MambaUKan(
            encoder_types=['Conv', 'EPA', 'EPA', 'KAN', 'KAN'],
            decoder_types=['Conv', 'EPA', 'EPA', 'KAN', 'KAN'],
            block_depths=3,
            down_sample_first=True,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network

class MambaAllTrainer(MambaUKanTrainer):
    """
    B
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 300
        self.initial_lr = 1e-3

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
                                                            n_stages=5,
                                                            print_args=True)

        network = MambaUKan(
            encoder_types=['Conv', 'MChannel', 'MChannel', 'MChannel', 'MChannel'],
            decoder_types=['Conv', 'MChannel', 'MChannel', 'MChannel', 'MChannel'],
            down_sample_first=True,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class ResMambaAllTrainer(MambaUKanTrainer):
    """
    B + D
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3

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
                                                            n_stages=5,
                                                            print_args=True)

        network = MambaUKan(
            encoder_types=['Conv', 'MChannel', 'MChannel', 'MChannel', 'MChannel'],
            decoder_types=['Conv', 'MChannel', 'MChannel', 'MChannel', 'MChannel'],
            down_sample_first=True,
            res_path_count=[4, 3, 2, 1],

            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network

class SFResUKanTrainer(MambaUKanTrainer):
    """
    A + C + D
    """

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
                                                            n_stages=5,
                                                            print_args=True)

        network = MambaUKan(
            res_path_count=[4, 3, 2, 1],
            select_fusion=True,
            skip_merge_type=None,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network

class SFEPAUKanTrainer(MambaUKanTrainer):
    """
    A + C + H
    """

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
        architecture_kwargs = HTTrainer.update_network_args(arch_init_kwargs, arch_init_kwargs_req_import,
                                                            num_input_channels, num_output_channels,
                                                            enable_deep_supervision,
                                                            n_stages=5,
                                                            print_args=True)

        network = MambaUKan(
            encoder_types=['Conv', 'EPA', 'EPA', 'KAN', 'KAN'],
            decoder_types=['Conv', 'EPA', 'EPA', 'KAN', 'KAN'],
            down_sample_first=True,
            select_fusion=True,
            skip_merge_type=None,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network

class SFMambaUKanTrainer(MambaUKanTrainer):
    """
    B + C
    """

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
                                                            n_stages=5,
                                                            print_args=True)

        network = MambaUKan(
            select_fusion=True,
            skip_merge_type=None,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network
