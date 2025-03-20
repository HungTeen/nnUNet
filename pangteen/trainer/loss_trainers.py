from typing import Union, List, Tuple

import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from pangteen.loss.combine_loss import DC_and_CE_and_Focal_loss
from pangteen.loss.uumamba import AutoWeighted_DC_and_CE_and_Focal_loss
from pangteen.network.ptnet.mamba_ukan import MambaUKan
from pangteen.trainer.trainers import HTTrainer


class AutoWeightedLossTrainer(HTTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    def configure_loss(self):
        return AutoWeighted_DC_and_CE_and_Focal_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {},
            {'alpha': 0.5, 'gamma': 2, 'smooth': 1e-5},
            ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)


class DiceCEFocalTrainer(HTTrainer):

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

    def configure_loss(self):
        return DC_and_CE_and_Focal_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {},
            {'alpha': 0.5, 'gamma': 2, 'smooth': 1e-5},
            ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)


class DiceTrainer(HTTrainer):

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

    def _build_loss(self):
        loss = MemoryEfficientSoftDiceLoss(**{'batch_dice': self.configuration_manager.batch_dice,
                                              'do_bg': self.label_manager.has_regions, 'smooth': 1e-5,
                                              'ddp': self.is_ddp},
                                           apply_nonlin=torch.sigmoid if self.label_manager.has_regions else softmax_helper_dim1)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
