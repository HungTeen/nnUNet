from typing import Union, List, Tuple

import torch
from torch import nn

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from pangteen.loss.combine_loss import DC_and_CE_and_Focal_loss
from pangteen.loss.uumamba import AutoWeighted_DC_and_CE_and_Focal_loss, AutoWeighted_DC_and_CE_loss
from pangteen.network.ptnet.mamba_ukan import SKIM_UNet
from pangteen.network.ptnet.ptukan import UKAN_3D, SFUKAN_3D
from pangteen.network.ukan.ukan_2d import UKAN
from pangteen.trainer.mamba_ukan_trainers import MambaUKanTrainer
from pangteen.trainer.trainers import HTTrainer

class ShiftedKANL1Trainer(HTTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500
        self.initial_lr = 1e-4

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

        network = SKIM_UNet(
            encoder_types=['XT', 'XT', 'XT', 'XT', 'XT', 'KAN'],
            decoder_types=['XT', 'XT', 'XT', 'XT', 'XT', 'KAN'],
            down_sample_first=True,
            skip_fusion=True,
            fusion_count=3,
            use_shift=True,
            no_kan=False,
            select_fusion=True,
            skip_merge_type=None,
            use_max=True,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class ShiftedKANL3Trainer(ShiftedKANL1Trainer):

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

        network = SKIM_UNet(
            encoder_types=['XT', 'XT', 'XT', 'KAN', 'KAN', 'KAN'],
            decoder_types=['XT', 'XT', 'XT', 'KAN', 'KAN', 'KAN'],
            down_sample_first=True,
            skip_fusion=True,
            fusion_count=3,
            use_shift=True,
            no_kan=False,
            select_fusion=True,
            skip_merge_type=None,
            use_max=True,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class ShiftedKANL4Trainer(ShiftedKANL1Trainer):

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

        network = SKIM_UNet(
            encoder_types=['XT', 'XT', 'KAN', 'KAN', 'KAN', 'KAN'],
            decoder_types=['XT', 'XT', 'KAN', 'KAN', 'KAN', 'KAN'],
            down_sample_first=True,
            skip_fusion=True,
            tri_orientation=True,
            fusion_count=3,
            use_shift=True,
            no_kan=False,
            select_fusion=True,
            skip_merge_type=None,
            use_max=True,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class ShiftedKANL5Trainer(ShiftedKANL1Trainer):

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

        network = SKIM_UNet(
            encoder_types=['XT', 'KAN', 'KAN', 'KAN', 'KAN', 'KAN'],
            decoder_types=['XT', 'KAN', 'KAN', 'KAN', 'KAN', 'KAN'],
            down_sample_first=True,
            skip_fusion=True,
            tri_orientation=True,
            fusion_count=3,
            use_shift=True,
            no_kan=False,
            select_fusion=True,
            skip_merge_type=None,
            use_max=True,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class MLPUKanTrainer(ShiftedKANL1Trainer):

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

        network = SKIM_UNet(
            encoder_types=['XT', 'XT', 'XT', 'XT', 'KAN', 'KAN'],
            decoder_types=['XT', 'XT', 'XT', 'XT', 'KAN', 'KAN'],
            down_sample_first=True,
            skip_fusion=True,
            tri_orientation=True,
            fusion_count=3,
            use_shift=False,
            no_kan=True,
            select_fusion=True,
            skip_merge_type=None,
            use_max=True,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class ShiftedMLPUKanTrainer(ShiftedKANL1Trainer):

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

        network = SKIM_UNet(
            encoder_types=['XT', 'XT', 'XT', 'XT', 'KAN', 'KAN'],
            decoder_types=['XT', 'XT', 'XT', 'XT', 'KAN', 'KAN'],
            down_sample_first=True,
            skip_fusion=True,
            tri_orientation=True,
            fusion_count=3,
            use_shift=True,
            no_kan=True,
            select_fusion=True,
            skip_merge_type=None,
            use_max=True,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class NoShiftedUKanTrainer(ShiftedKANL1Trainer):

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

        network = SKIM_UNet(
            encoder_types=['XT', 'XT', 'XT', 'XT', 'KAN', 'KAN'],
            decoder_types=['XT', 'XT', 'XT', 'XT', 'KAN', 'KAN'],
            down_sample_first=True,
            skip_fusion=True,
            tri_orientation=True,
            fusion_count=3,
            use_shift=False,
            no_kan=False,
            select_fusion=True,
            skip_merge_type=None,
            use_max=True,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network