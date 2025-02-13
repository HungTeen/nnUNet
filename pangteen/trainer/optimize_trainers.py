import torch

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from pangteen.loss.uumamba import SAM
from pangteen.trainer.trainers import HTTrainer


class SAMOptimizerTrainer(HTTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    def configure_optimizers(self):
        param_groups = [
            {'params': self.network.parameters(), 'weight_decay': self.weight_decay, 'lr': self.initial_lr},
            {'params': self.loss.awl.parameters(), 'weight_decay': 0, 'lr': self.initial_lr}
        ]
        base_optimizer = torch.optim.SGD
        optimizer = SAM(param_groups, base_optimizer, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler