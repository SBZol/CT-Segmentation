#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   loss_function.py
@Time    :   2021/05/22 21:49:50
@Author  :   Dio
@Version :   1.0
@Contact :   sbzol.chen@gmail.com
@License :   None
@Desc    :   None
'''

# here put the import lib
import torch
from torch.autograd import Function


class DiceCoeff(Function):
    """Dice coeff for individual examples"""
    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def dice_ceoff1(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)
    m2 = target.view(num, -1)
    intersection = (m1 / m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
