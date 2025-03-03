import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution
from monai.networks.layers import Act, Norm
import pdb

# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


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


def dynamic_kd_loss(student_logits, teacher_logits, temperature=3.0) -> torch.Tensor:
    loss = 0.
    with torch.no_grad():
        student_probs = torch.sigmoid(student_logits)
        teacher_probs = torch.sigmoid(teacher_logits)
        student_entropy = - teacher_probs * torch.log(student_probs + 1e-10)  # student entropy, (bsz, )
        student_entropy += - (1 - teacher_probs) * torch.log((1 - student_probs) + 1e-10)  # student entropy, (bsz, )
        # normalized entropy score by student uncertainty:
        # i.e.,  entropy / entropy_upper_bound
        # higher uncertainty indicates the student is more confusing about this instance
        instance_weight = student_entropy / torch.max(student_entropy)

    batch_loss = kl_div(student_logits / temperature, teacher_logits / temperature) * temperature ** 2
    loss += torch.mean(batch_loss * torch.cat([instance_weight, instance_weight], dim=1))
    return loss


def kd_loss_f(student_logits, teacher_logits, temperature=3.0) -> torch.Tensor:
    loss = 0.
    batch_loss = kl_div(student_logits / temperature, teacher_logits / temperature) * temperature ** 2
    loss += torch.mean(batch_loss)
    return loss

class UncertaintyTeacherKDForSequenceClassification(nn.Module):
    def __init__(self,
                 kd_alpha=0.5,
                 ce_alpha=0.5,
                 en_alpha=0.,
                 t=0.9,
                 loss_func=None,
                 temperature=5.0,
                 student=None,
                 ende="en",
                 dy_loss: bool = True
                 ):
        super().__init__()
        self.student = student
        self.kd_alpha = kd_alpha
        self.ce_alpha = ce_alpha
        self.en_alpha = en_alpha
        self.temperature = temperature
        self.loss_func = loss_func
        self.ende = ende
        self.dy_loss = dy_loss
        self.t = t

    def forward(self, inputs=None, labels=None, teacher_logits=None):
        loss = 0.
        if self.training:
            student_logits = self.student(inputs)
        else:
            student_logits = self.student(inputs)
            return student_logits
        if self.dy_loss:
            kd_loss = dynamic_kd_loss(student_logits, teacher_logits, self.temperature)
        else:
            kd_loss = kd_loss_f(student_logits, teacher_logits, self.temperature)
        entropy_loss = cross_entropy(student_logits, torch.sigmoid(teacher_logits), self.t).mean()
        dice_loss = self.loss_func(student_logits, labels)
        if self.en_alpha != 0.:
            loss += self.en_alpha * entropy_loss
        loss += self.ce_alpha * dice_loss
        if teacher_logits is not None:
            loss += self.kd_alpha * kd_loss
        return loss, kd_loss, dice_loss, entropy_loss

    def mock_foward(self, student, teacher):
        if self.dy_loss:
            kd_loss = dynamic_kd_loss(student, teacher, self.temperature)
        else:
            kd_loss = kd_loss_f(student, teacher, self.temperature)
        return kd_loss

class DynamicWeightKD(nn.Module):

    def __int__(self, temperature=3.0):
        super(DynamicWeightKD, self).__init__()
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        return dynamic_kd_loss(student_logits, teacher_logits, self.temperature)

if __name__ == "__main__":
    loss_func = UncertaintyTeacherKDForSequenceClassification(kd_alpha=0.5, ce_alpha=0.5, en_alpha=0., t=0.9, loss_func=None, temperature=5.0, student=None, ende="en", dy_loss=False)
    x = torch.randn(1, 3, 128, 128, 128)
    y = torch.randn(1, 3, 128, 128, 128)
    loss = loss_func.mock_foward(x, y)
    print("Random Loss: ", loss)
    x = torch.ones(1, 3, 128, 128, 128)
    y = torch.ones(1, 3, 128, 128, 128)
    loss = loss_func.mock_foward(x, y)
    print("Ones Loss: ", loss)
    x = torch.zeros(1, 3, 128, 128, 128)
    loss = loss_func.mock_foward(x, y)
    print("Zeros Loss: ", loss)