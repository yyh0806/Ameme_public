import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from loss.losses import *


def CrossEntropyLoss(output, target):
    return F.cross_entropy(output, target)


def FocalCosineLoss(output, target):
    reduction = "mean"
    cosine_loss = F.cosine_embedding_loss(output, F.one_hot(target, num_classes=output.size(-1)),
                                          torch.Tensor([1]).cuda(), reduction=reduction)
    cent_loss = F.cross_entropy(F.normalize(output), target, reduce=False)
    pt = torch.exp(-cent_loss)
    focal_loss = 1 * (1 - pt) ** 2 * cent_loss

    if reduction == "mean":
        focal_loss = torch.mean(focal_loss)

    return cosine_loss + 0.1 * focal_loss


def BiTemperedLoss(output, target):
    loss_function = BiTemperedLogisticLoss(reduction='mean', t1=0.6, t2=1.4, label_smoothing=0.2).cuda()
    return loss_function(output, target)