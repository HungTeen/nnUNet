import torch
import torch.nn as nn

from nnunetv2.training.loss.dice import SoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from pangteen.loss.focal_loss import FocalLoss


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
#         for param in self.parameters():
#             print(param)
        return loss_sum

########## Uncertainty-Aware loss (DC + CE + Focal) ##########
class AutoWeighted_DC_and_CE_and_Focal_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, focal_kwargs, ignore_label=None, dice_class=SoftDiceLoss):
        """
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param focal_kwargs:
        """
        super(AutoWeighted_DC_and_CE_and_Focal_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.ignore_label = ignore_label

        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.focal = FocalLoss(apply_nonlin=softmax_helper_dim1, **focal_kwargs)
        self.awl = AutomaticWeightedLoss(3)

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
        result = self.awl(dc_loss, ce_loss, focal_loss)
        return result


########## Uncertainty-Aware loss (DC + CE) ##########
class AutoWeighted_DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, ignore_label=None, dice_class=SoftDiceLoss):

        super(AutoWeighted_DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.ignore_label = ignore_label

        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.awl = AutomaticWeightedLoss(2)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):

        if self.ignore_label is not None:
            assert target.shape[
                       1] == 1, 'ignore label is not implemented for one hot encoded target variables (DC_and_CE_loss)'
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask)
        ce_loss = self.ce(net_output, target[:, 0])
        result = self.awl(dc_loss, ce_loss)
        return result


########## Uncertainty-Aware loss (CE + Focal) ##########
class AutoWeighted_CE_and_Focal_loss(nn.Module):
    def __init__(self, ce_kwargs, focal_kwargs, ignore_label=None):
        super(AutoWeighted_CE_and_Focal_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.focal = FocalLoss(apply_nonlin=softmax_helper_dim1, **focal_kwargs)
        self.awl = AutomaticWeightedLoss(2)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        target_focal = target[:, 0].long()

        ce_loss = self.ce(net_output, target[:, 0])
        focal_loss = self.focal(net_output, target_focal)
        result = self.awl(ce_loss, focal_loss)

        return result


########## Uncertainty-Aware loss (DC + topk) ##########
class AutoWeighted_DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, topk_kwargs, ignore_label=None, dice_class=SoftDiceLoss):

        super(AutoWeighted_DC_and_topk_loss, self).__init__()
        if ignore_label is not None:
            topk_kwargs['ignore_index'] = ignore_label

        self.ignore_label = ignore_label

        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.topk = TopKLoss(**topk_kwargs)
        self.awl = AutomaticWeightedLoss(2)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):

        if self.ignore_label is not None:
            assert target.shape[
                       1] == 1, 'ignore label is not implemented for one hot encoded target variables (DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask)
        topk_loss = self.topk(net_output, target)
        result = self.awl(dc_loss, topk_loss)
        return result


########## Uncertainty-Aware loss (CE + topk) ##########
class AutoWeighted_CE_and_topk_loss(nn.Module):
    def __init__(self, ce_kwargs, topk_kwargs, ignore_label=None):
        super(AutoWeighted_CE_and_topk_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.topk = TopKLoss(**topk_kwargs)
        self.awl = AutomaticWeightedLoss(2)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        ce_loss = self.ce(net_output, target[:, 0])
        topk_loss = self.topk(net_output, target)
        result = self.awl(ce_loss, topk_loss)

        return result