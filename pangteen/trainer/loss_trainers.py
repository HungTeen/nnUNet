import torch

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from pangteen.loss.uumamba import SAM, AutoWeighted_DC_and_CE_loss, AutoWeighted_DC_and_CE_and_Focal_loss
from pangteen.trainer.trainers import HTTrainer


class AutoWeightedLossTrainer(HTTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    def configure_loss(self):
        return AutoWeighted_DC_and_CE_and_Focal_loss({'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                                           {},
                                                           {'alpha':0.5, 'gamma':2, 'smooth':1e-5},
                                                           ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)