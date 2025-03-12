from typing import Union, List, Tuple

import torch
from torch import nn

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from pangteen.loss.uumamba import AutoWeighted_DC_and_CE_and_Focal_loss, AutoWeighted_DC_and_CE_loss, \
    AutoWeighted_CE_and_Focal_loss
from pangteen.network.ptnet.mamba_ukan import MambaUKan
from pangteen.trainer.ptukan_trainers import PTUKanTrainer
from pangteen.trainer.trainers import HTTrainer


class SFUKanWithLossTrainer(PTUKanTrainer):
    """
    A + B + L
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

        network = MambaUKan(
            encoder_types=['Conv', 'Conv', 'Conv', 'Conv', 'KAN', 'KAN'],
            decoder_types=['Conv', 'Conv', 'Conv', 'Conv', 'KAN', 'KAN'],
            select_fusion=True,
            skip_merge_type=None,
            use_max=True,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network

class DiceCEAndFocalTrainer(SFUKanWithLossTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                     device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    def configure_loss(self):
        return AutoWeighted_DC_and_CE_and_Focal_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {},
            {'alpha': 0.5, 'gamma': 2, 'smooth': 1e-5},
            ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)


class DiceAndCETrainer(SFUKanWithLossTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    def configure_loss(self):
        return AutoWeighted_DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {},
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss)


class DiceAndFocalTrainer(SFUKanWithLossTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    def configure_loss(self):
        return AutoWeighted_CE_and_Focal_loss(
            {},
            {'alpha': 0.5, 'gamma': 2, 'smooth': 1e-5},
            ignore_label=self.label_manager.ignore_label)