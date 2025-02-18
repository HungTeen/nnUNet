import torch
import torch.nn.functional as F
from torch import nn


def cross_entropy(x, y, t=0.9):
    student_probs = torch.sigmoid(x)
    student_entropy = - y * torch.log(student_probs + 1e-10)  # student entropy, (bsz, )
    _y = torch.ones_like(y)
    _y[y >= t] = 0.
    student_entropy += - _y * torch.log((1 - student_probs) + 1e-10)

    return student_entropy


def kl_div(x, y):
    input = (torch.sigmoid(x) + 1e-10)
    input = torch.cat([input, 1 - input], dim=1)
    target = torch.sigmoid(y)
    target = torch.cat([target, 1 - target], dim=1)
    kl = F.kl_div(input, target, reduction="none", log_target=True)
    return kl
#
# def dynamic_kd_loss(student_logits, teacher_logits, temperature=3.0) -> torch.Tensor:
#     loss = 0.
#     with torch.no_grad():
#         student_probs = torch.sigmoid(student_logits)
#         teacher_probs = torch.sigmoid(teacher_logits)
#         student_entropy = - teacher_probs * torch.log(student_probs + 1e-10)  # student entropy, (bsz, )
#         student_entropy += - (1 - teacher_probs) * torch.log((1 - student_probs) + 1e-10)  # student entropy, (bsz, )
#         # normalized entropy score by student uncertainty:
#         # i.e.,  entropy / entropy_upper_bound
#         # higher uncertainty indicates the student is more confusing about this instance
#         instance_weight = student_entropy / torch.max(student_entropy)
#
#     batch_loss = kl_div(student_logits / temperature, teacher_logits / temperature) * temperature ** 2
#     loss += torch.mean(batch_loss * torch.cat([instance_weight, instance_weight], dim=1))
#     return loss

def dynamic_kd_loss(student_logits, teacher_logits, temperature=3.0) -> torch.Tensor:
    loss = 0.
    with torch.no_grad():
        student_probs = torch.sigmoid(student_logits)
        teacher_probs = torch.sigmoid(teacher_logits)

        # 避免 log(0) 造成 NaN
        student_probs = torch.clamp(student_probs, min=1e-10, max=1.0)
        teacher_probs = torch.clamp(teacher_probs, min=1e-10, max=1.0)

        student_entropy = - teacher_probs * torch.log(student_probs)  # (bsz,)
        student_entropy += - (1 - teacher_probs) * torch.log(1 - student_probs)

        # 避免除零错误
        instance_weight = student_entropy / (torch.max(student_entropy) + 1e-10)
        instance_weight = instance_weight.unsqueeze(1)  # 变成 (batch_size, 1)

    # 避免 KL 散度出现 NaN
    batch_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='none'
    ) * (temperature ** 2)

    # 维度对齐
    loss += torch.mean(batch_loss * torch.cat([instance_weight, instance_weight], dim=1))
    return loss

def kd_loss_f(student_logits, teacher_logits, temperature=3.0) -> torch.Tensor:
    loss = 0.
    batch_loss = kl_div(student_logits / temperature, teacher_logits / temperature) * temperature ** 2
    loss += torch.mean(batch_loss)
    return loss


# class CriterionCWD(nn.Module):
#
#     def __init__(self, divergence='mse', temperature=1.0):
#
#         super(CriterionCWD, self).__init__()
#
#         self.normalize = nn.Softmax(dim=1)
#         self.temperature = 1.0
#
#         # define loss function
#         if divergence == 'mse':
#             self.criterion = nn.MSELoss(reduction='mean')
#         elif divergence == 'kl':
#             self.criterion = nn.KLDivLoss(reduction='mean')
#             self.temperature = temperature
#         self.divergence = divergence
#
#     def forward(self, preds_S, preds_T):
#         norm_s = self.normalize(preds_S / self.temperature)
#         norm_t = self.normalize(preds_T.detach() / self.temperature)
#
#         if self.divergence == 'kl':
#             norm_s = norm_s.log()
#         loss = self.criterion(norm_s, norm_t)
#
#         # # loss 除以像素点个数。
#         # loss /= preds_S.shape[2] * preds_S.shape[3] * preds_S.shape[4]
#
#         return loss * (self.temperature ** 2)


class UncertaintyTeacherKDForSequenceClassification(nn.Module):
    def __init__(self,
                 loss_func=None,
                 kd_alpha=0.5,
                 temperature=5.0,
                 dy_loss: bool = False
                 ):
        super().__init__()
        self.kd_alpha = kd_alpha
        self.temperature = temperature
        self.loss_func = loss_func
        self.dy_loss = dy_loss

    def forward(self, student_output, target, teacher_output=None):
        if self.loss_func is None:
            loss = 0.
        else:
            loss = self.loss_func(student_output, target)

        if teacher_output is None:
            return loss

        if isinstance(student_output, list):
            student_output = student_output[-1]
        if isinstance(teacher_output, list):
            teacher_output = teacher_output[-1]

        kd_loss = 0.
        if self.dy_loss:
            kd_loss = dynamic_kd_loss(student_output, teacher_output, self.temperature)
        else:
            kd_loss = kd_loss_f(student_output, teacher_output, self.temperature)

        if teacher_output is not None:
            loss += self.kd_alpha * kd_loss
        return loss, kd_loss


import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivergenceLoss3D(nn.Module):
    def __init__(self, reduction='mean', eps=1e-6):
        """
        三维KL散度损失函数
        Args:
            reduction (str): 损失聚合方式，可选 'mean'(默认)/'sum'/'batchmean'(数学KL公式对齐)
            eps (float): 数值稳定项（避免log(0)）
        """
        super().__init__()
        assert reduction in ['mean', 'sum', 'batchmean'], "reduction参数需为mean/sum/batchmean"
        self.reduction = reduction
        self.eps = eps

    def forward(self, input, target):
        """
        Args:
            input (Tensor):  模型输出的logits (未归一化)，维度 [B, C, D, H, W]
            target (Tensor): 目标分布，维度为：
                             1. [B, C, D, H, W] (概率分布)
                             或
                             2. [B, 1, D, H, W]/[B, D, H, W] (类别标签)
        """
        # Step 1: 检查输入维度
        assert input.dim() == 5, "输入应为5D张量: [B, C, D, H, W]"
        B, C = input.size(0), input.size(1)
        spatial_dims = input.shape[2:]

        # Step 2: 处理目标张量
        if target.dim() == 5 and target.size(1) == C:
            # Case 1: 目标为现成的概率分布
            target_probs = target
        else:
            # Case 2: 目标为类别标签 → 转为one-hot编码
            if target.dim() == 4:
                target = target.unsqueeze(1)  # [B, 1, D, H, W]
            assert target.size(1) == 1, "标签张量的通道数应为1"

            # 生成one-hot编码 [B, C, D, H, W]
            target_onehot = torch.zeros(B, C, *spatial_dims,
                                        dtype=input.dtype, device=input.device)
            target_onehot.scatter_(1, target.long(), 1)  # 填充1到对应类别位置
            target_probs = target_onehot

        # Step 3: 数值稳定处理
        input_log_probs = F.log_softmax(input, dim=1)  # 对输入计算log概率
        target_probs = torch.clamp(target_probs, min=self.eps)  # 避免概率为0

        # Step 4: 计算KL散度 (sum_target(p * (log(p) - log(q))))
        kl_div = F.kl_div(
            input=input_log_probs,  # 需为 log-probabilities
            target=target_probs,  # 需为 probabilities
            reduction='none'  # 后续手动处理reduction
        )  # 输出维度 [B, C, D, H, W]

        # Step 5: 聚合损失 (沿所有维度求和，然后按reduction聚合)
        if self.reduction == 'mean':
            loss = torch.mean(kl_div)
        elif self.reduction == 'sum':
            loss = torch.sum(kl_div)
        elif self.reduction == 'batchmean':  # 对齐数学公式定义 (总项数为B)
            loss = torch.sum(kl_div) / B

        return loss


class KLDivergenceKD3D(nn.Module):
    def __init__(self, temperature=1.0, reduction='batchmean', eps=1e-8):
        """
        3D 语义分割 知识蒸馏 KL 散度损失函数
        Args:
            temperature (float): 蒸馏温度 T，默认 1.0
            reduction (str): 'mean' | 'sum' | 'batchmean'（默认）
            eps (float): 避免 log(0) 计算错误
        """
        super().__init__()
        assert reduction in ['mean', 'sum', 'batchmean'], "reduction 需为 'mean'/'sum'/'batchmean'"
        self.temperature = temperature
        self.reduction = reduction
        self.eps = eps

    def forward(self, student_logits, teacher_logits):
        """
        计算 Student 网络和 Teacher 网络的 KL 散度
        Args:
            student_logits (Tensor): Student 模型的 logits，形状 [B, C, D, H, W]
            teacher_logits (Tensor): Teacher 模型的 logits，形状 [B, C, D, H, W]
        Returns:
            loss (Tensor): 计算得到的 KL 散度损失
        """
        assert student_logits.shape == teacher_logits.shape, "Student 和 Teacher 的输出形状必须相同"

        T = self.temperature  # 取温度

        # 计算 softmax 后的概率分布
        student_probs = F.log_softmax(student_logits / T, dim=1)  # log softmax
        teacher_probs = F.softmax(teacher_logits / T, dim=1)  # softmax

        # 避免 log(0) 计算错误
        teacher_probs = torch.clamp(teacher_probs, min=self.eps)

        # 计算 KL 散度
        kl_loss = F.kl_div(student_probs, teacher_probs, reduction='none')  # [B, C, D, H, W]

        # 按温度 T 进行缩放
        kl_loss = kl_loss * (T ** 2)

        # 计算最终的损失
        if self.reduction == 'mean':
            loss = kl_loss.mean()
        elif self.reduction == 'sum':
            loss = kl_loss.sum()
        elif self.reduction == 'batchmean':  # 对齐 KL 数学定义
            loss = kl_loss.sum() / student_logits.size(0)

        return loss


class KDLoss(nn.Module):
    def __init__(self,
                 loss_func=None,
                 kd_alpha=0.5,
                 temperature=3.0,
                 dy_loss: bool = False
                 ):
        super().__init__()
        self.kd_alpha = kd_alpha
        self.temperature = temperature
        self.loss_func = loss_func
        # self.kd_loss = nn.KLDivLoss(reduction='mean')
        # self.kd_loss = KLDivergenceLoss3D()
        self.kd_loss = KLDivergenceKD3D(reduction='mean')

    def forward(self, student_output, target, teacher_output=None):
        if self.loss_func is None:
            loss = 0.
        else:
            loss = self.loss_func(student_output, target)

        if teacher_output is not None:
            if not isinstance(student_output, list):
                student_output = [student_output]
            if not isinstance(teacher_output, list):
                teacher_output = [teacher_output]
            for i in range(min(len(teacher_output), len(student_output))):
                if self.kd_alpha > 0:
                    stu = student_output[i]
                    tea = teacher_output[i]
                    loss += self.kd_alpha * self.kd_loss(stu, tea)
        return loss


if __name__ == '__main__':
    student_logits = torch.randn(2, 2, 2, 2, 2)
    teacher_logits = torch.randn(2, 2, 2, 2, 2)
    loss = KLDivergenceKD3D(reduction='mean')(student_logits, teacher_logits)
    print(loss)
    student_logits = torch.randn(2, 2, 2, 2, 2)
    teacher_logits = torch.zeros(2, 2, 2, 2, 2)
    loss = KLDivergenceKD3D(reduction='mean')(student_logits, teacher_logits)
    print(loss)