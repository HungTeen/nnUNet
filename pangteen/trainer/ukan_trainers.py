from typing import Union, List, Tuple

import torch
from torch import nn

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from pangteen.loss.uumamba import AutoWeighted_DC_and_CE_and_Focal_loss, AutoWeighted_DC_and_CE_loss
from pangteen.network.ptnet.ptukan import UKAN_3D, SFUKAN_3D
from pangteen.network.ukan.ukan_2d import UKAN
from pangteen.trainer.trainers import HTTrainer


class UKanTrainer(HTTrainer):
    """
    CUDA_VISIBLE_DEVICES=2 nohup python -u pangteen/train.py 201 3d_fullres 3 -p stage5Plans -tr UKanTrainer -num_gpus 1 > main0.out 2>&1 &
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500
        self.initial_lr = 1e-2
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

        network = UKAN_3D(
            encode_kan_num=2,
            decode_kan_num=2,
            reverse_kan_order=True,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class SFUKanTrainer(UKanTrainer):
    """
    SF + UKan
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
                                                            print_args=True,
                                                            n_stages=5
                                                            )

        network = SFUKAN_3D(
            encode_kan_num=2,
            decode_kan_num=2,
            reverse_kan_order=True,
            embed_dims=[32, 64, 128, 256, 512],
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class Weighted_SFUKanTrainer(SFUKanTrainer):
    """
    验证Loss+SFUKan网络的训练器。
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    def configure_loss(self):
        return AutoWeighted_DC_and_CE_and_Focal_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {},
            {'alpha': 0.5, 'gamma': 2, 'smooth': 1e-5},
            ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)


class Weighted_Dice_CE_SFUKanTrainer(SFUKanTrainer):
    """
    验证Loss+SFUKan网络的训练器。
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    def configure_loss(self):
        return AutoWeighted_DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {},
            {},
            dice_class=MemoryEfficientSoftDiceLoss)

class SmallUKanTrainer(UKanTrainer):

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

        network = UKAN_3D(
            encode_kan_num=2,
            decode_kan_num=2,
            reverse_kan_order=True,
            embed_dims=[32, 64, 128, 256, 320],
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network

class UKan2DTrainer(HTTrainer):
    """
    CUDA_VISIBLE_DEVICES=2 nohup python -u pangteen/train.py 201 2d 3 -tr UKan2DTrainer -num_gpus 1 > main0.out 2>&1 &
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100
        self.initial_lr = 1e-2
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
                                                            print_args=True)

        network = UKAN(
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network