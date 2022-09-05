import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Optional


class Lossncriterion(nn.Module):
    def __init__(self):
        super(Lossncriterion, self).__init__()
        self.laplacian = torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], dtype=torch.half).requires_grad_(False).cuda()
        self.converter = nn.Sigmoid()
        
    def dice_coefficient(self, inputs, targets):
        inputs = torch.nn.Sigmoid()(inputs)
        inputs = (inputs > 0.5).float()
        if targets.sum() == 0:
            if inputs.sum() == 0:
                return torch.tensor(1., device="cuda")
            else:
                return torch.tensor(0., device="cuda")
        intersection = torch.sum(inputs * targets)
        dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        return dice
    
    def binary_dice(self, inputs, targets):
        smooth = 1.
        intersection = torch.sum(inputs * targets)
        dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)
        return 1 - dice
        
    def forward(self, inputs, target):
        wbce, wiou = structure_loss(inputs, target)
        loss = wbce.mean() + wiou.mean()
        return loss

    def boundary_forward(self, pred, gt):
        pred = nn.Sigmoid()(pred)
        gt = (F.conv2d(gt.float(), self.laplacian, stride=1, padding = 1) > 0.1).float()
        bce_loss = F.binary_cross_entropy_with_logits(pred, gt.expand(gt.size(0), pred.size(1), gt.size(2), gt.size(3)))
        return bce_loss

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = (pred*weit).sum(dim=(2, 3)) + (mask*weit).sum(dim=(2, 3))  #((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter + 1)
    return wbce, wiou
