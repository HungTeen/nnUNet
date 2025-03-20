########## Uncertainty-Aware loss (DC + CE + Focal) ##########
import torch
from torch import nn

from nnunetv2.training.loss.dice import SoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from pangteen.loss.focal_loss import FocalLoss


class DC_and_CE_and_Focal_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, focal_kwargs, ignore_label=None, dice_class=SoftDiceLoss):
        """
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param focal_kwargs:
        """
        super(DC_and_CE_and_Focal_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.ignore_label = ignore_label

        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.focal = FocalLoss(apply_nonlin=softmax_helper_dim1, **focal_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[
                       1] == 1, 'ignore label is not implemented for one hot encoded target variables (AutoWeighted_DC_and_CE_and_Focal_loss)'
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        target_focal = target[:, 0].long()

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask)
        ce_loss = self.ce(net_output, target[:, 0])
        focal_loss = self.focal(net_output, target_focal)
        result = dc_loss + ce_loss + focal_loss
        return result