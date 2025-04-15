from typing import Union, List, Tuple

import torch
from torch import nn

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from pangteen.loss.uumamba import AutoWeighted_DC_and_CE_and_Focal_loss, AutoWeighted_DC_and_CE_loss, \
    AutoWeighted_CE_and_Focal_loss
from pangteen.network.ptnet.mamba_ukan import SKIM_UNet
from pangteen.trainer.ptukan_trainers import PTUKanTrainer
from pangteen.trainer.skim_trainers import SKIMUNetTrainer
from pangteen.trainer.trainers import HTTrainer


class SKIMLossTrainer(SKIMUNetTrainer):
    """
    A + B + L
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500
        self.initial_lr = 1e-2

class DiceTrainer(SKIMLossTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                     device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    def configure_loss(self):
        return MemoryEfficientSoftDiceLoss(**{'batch_dice': self.configuration_manager.batch_dice,
                                              'do_bg': self.label_manager.has_regions, 'smooth': 1e-5,
                                              'ddp': self.is_ddp},
                                           apply_nonlin=torch.sigmoid if self.label_manager.has_regions else softmax_helper_dim1)


class DiceCEAndFocalTrainer(SKIMLossTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                     device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    def configure_loss(self):
        return AutoWeighted_DC_and_CE_and_Focal_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {},
            {'alpha': 0.5, 'gamma': 2, 'smooth': 1e-5},
            ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)


class DiceAndCETrainer(SKIMLossTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    def configure_loss(self):
        return AutoWeighted_DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {},
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss)


class DiceAndFocalTrainer(SKIMLossTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    def configure_loss(self):
        return AutoWeighted_CE_and_Focal_loss(
            {},
            {'alpha': 0.5, 'gamma': 2, 'smooth': 1e-5},
            ignore_label=self.label_manager.ignore_label)