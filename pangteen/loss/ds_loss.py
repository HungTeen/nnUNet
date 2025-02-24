import torch
import torch.nn.functional as F
from torch import nn

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