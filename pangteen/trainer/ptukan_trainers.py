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


class PTUKanTrainer(HTTrainer):
    """
    A: KAN | B: SF | C: Mamba Encode | D: ResPath | E: GSC | F: EPA | G: MF
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

        network = SKIM_UNet(
            encoder_types=['Conv', 'Conv', 'Conv', 'KAN', 'KAN'],
            decoder_types=['Conv', 'Conv', 'Conv', 'KAN', 'KAN'],
            down_sample_first=True,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class SFResUKanTrainer(MambaUKanTrainer):
    """
    A + B
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

        network = SKIM_UNet(
            select_fusion=True,
            skip_merge_type=None,
            encoder_types=['Conv', 'Conv', 'Conv', 'KAN', 'KAN'],
            decoder_types=['Conv', 'Conv', 'Conv', 'KAN', 'KAN'],
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class MSpatialUKanTrainer(PTUKanTrainer):
    """
    A + C
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

        network = SKIM_UNet(
            encoder_types=['Conv', 'MSpatial', 'MSpatial', 'KAN', 'KAN'],
            decoder_types=['Conv', 'MSpatial', 'MSpatial', 'KAN', 'KAN'],
            down_sample_first=True,
            mamba_count=2,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network

class ResUKanTrainer(PTUKanTrainer):
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

        network = SKIM_UNet(
            res_path_count=[4, 3, 2, 1],
            encoder_types=['Conv', 'Conv', 'Conv', 'KAN', 'KAN'],
            decoder_types=['Conv', 'Conv', 'Conv', 'KAN', 'KAN'],
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class GSCUKanTrainer(PTUKanTrainer):
    """
    A + E
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

        network = SKIM_UNet(
            feature_channels=[32, 64, 128, 256, 512],
            encoder_types=['Conv', 'GSC', 'GSC', 'KAN', 'KAN'],
            decoder_types=['Conv', 'GSC', 'GSC', 'KAN', 'KAN'],
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network

class EPAUKanTrainer(PTUKanTrainer):
    """
    A + F
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
                                                            cut_from_last=False,
                                                            print_args=True)

        network = SKIM_UNet(
            feature_channels=[32, 64, 128, 256, 512],
            encoder_types=['Conv', 'EPA', 'EPA', 'KAN', 'KAN'],
            decoder_types=['Conv', 'EPA', 'EPA', 'KAN', 'KAN'],
            block_depths=3,
            down_sample_first=True,
            upsample_last=True,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class MambaFusionUKanTrainer(PTUKanTrainer):
    """
    A + G
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
                                                            print_args=True)

        network = SKIM_UNet(
            skip_fusion=True,
            encoder_types=['Conv', 'Conv', 'Conv', 'Conv', 'KAN', 'KAN'],
            decoder_types=['Conv', 'Conv', 'Conv', 'Conv', 'KAN', 'KAN'],
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network

class XTUKanTrainer(PTUKanTrainer):
    """
    A + T
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
                                                            print_args=True)

        network = SKIM_UNet(
            encoder_types=['XT', 'XT', 'XT', 'XT', 'KAN', 'KAN'],
            decoder_types=['XT', 'XT', 'XT', 'XT', 'KAN', 'KAN'],
            down_sample_first=True,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network

class SFResUKanTrainer(MambaUKanTrainer):
    """
    A + B + D
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

        network = SKIM_UNet(
            res_path_count=[4, 3, 2, 1],
            select_fusion=True,
            skip_merge_type=None,
            encoder_types=['Conv', 'Conv', 'Conv', 'KAN', 'KAN'],
            decoder_types=['Conv', 'Conv', 'Conv', 'KAN', 'KAN'],
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class SFEPAUKanTrainer(MambaUKanTrainer):
    """
    A + B + F
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

        network = SKIM_UNet(
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


class SFMambaFusionUKanTrainer(PTUKanTrainer):
    """
    A + B + G
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 300
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
            skip_fusion=True,
            encoder_types=['Conv', 'Conv', 'Conv', 'Conv', 'KAN', 'KAN'],
            decoder_types=['Conv', 'Conv', 'Conv', 'Conv', 'KAN', 'KAN'],
            select_fusion=True,
            skip_merge_type=None,
            fusion_count=3,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class SFMFUKanV2Trainer(PTUKanTrainer):
    """
    A + B + G
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 300
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
            skip_fusion=True,
            encoder_types=['Conv', 'Conv', 'Conv', 'Conv', 'KAN', 'KAN'],
            decoder_types=['Conv', 'Conv', 'Conv', 'Conv', 'KAN', 'KAN'],
            select_fusion=True,
            skip_merge_type=None,
            tri_orientation=True,
            use_max=True,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class SFMFUKanV3Trainer(PTUKanTrainer):
    """
    A + B + G
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 300
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
            skip_fusion=True,
            encoder_types=['Conv', 'Conv', 'Conv', 'Conv', 'KAN', 'KAN'],
            decoder_types=['Conv', 'Conv', 'Conv', 'Conv', 'KAN', 'KAN'],
            select_fusion=True,
            skip_merge_type=None,
            tri_orientation=True,
            res_mamba_fusion=True,
            use_max=True,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network

class SFXTUKanTrainer(PTUKanTrainer):
    """
    A + B + T
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500
        self.initial_lr = 1e-2

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
            select_fusion=True,
            encoder_types=['XT', 'XT', 'XT', 'XT', 'KAN', 'KAN'],
            decoder_types=['XT', 'XT', 'XT', 'XT', 'KAN', 'KAN'],
            down_sample_first=True,
            skip_merge_type=None,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class MambaXTUKanTrainer(PTUKanTrainer):
    """
    A + G + T
    """

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
            encoder_types=['XT', 'XT', 'XT', 'XT', 'KAN', 'KAN'],
            decoder_types=['XT', 'XT', 'XT', 'XT', 'KAN', 'KAN'],
            skip_fusion=True,
            down_sample_first=True,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network

class SFUKanL1Trainer(PTUKanTrainer):
    """
    三层 KAN。
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500
        self.initial_lr = 1e-2

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
            encoder_types=['Conv', 'Conv', 'Conv', 'Conv', 'Conv', 'KAN'],
            decoder_types=['Conv', 'Conv', 'Conv', 'Conv', 'Conv', 'KAN'],
            select_fusion=True,
            skip_merge_type=None,
            use_max=True,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class SFUKanL3Trainer(PTUKanTrainer):
    """
    三层 KAN。
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500
        self.initial_lr = 1e-2

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
            encoder_types=['Conv', 'Conv', 'Conv', 'KAN', 'KAN', 'KAN'],
            decoder_types=['Conv', 'Conv', 'Conv', 'KAN', 'KAN', 'KAN'],
            select_fusion=True,
            skip_merge_type=None,
            use_max=True,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class SFUKanL4Trainer(PTUKanTrainer):
    """
    4 层 KAN。
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500
        self.initial_lr = 1e-2

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
            encoder_types=['Conv', 'Conv', 'KAN', 'KAN', 'KAN', 'KAN'],
            decoder_types=['Conv', 'Conv', 'KAN', 'KAN', 'KAN', 'KAN'],
            select_fusion=True,
            skip_merge_type=None,
            use_max=True,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network
