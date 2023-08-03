import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Optional

def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    # j = 0
    # for batch in gt:
    #     i = 0
    #     print(j, " = ", batch.shape)
    #     for ch in batch:
    #         has_ones = torch.any(ch == 1)
    #         if has_ones:
    #             print(i, " = YES")
    #         else:
    #             print(i, " = NO")
    #         i += 1
    #     j += 1
    
    pos_inds = gt.eq(1).float()
    pos_inds[pos_inds.isnan()] = 0
    neg_inds = gt.lt(1).float()
    neg_inds[neg_inds.isnan()] = 0
    
    # j = 0
    # for batch in neg_inds:
    #     i = 0
    #     print(j, " = ", batch.shape)
    #     for ch in batch:
    #         has_ones = torch.any(ch == 0)
    #         if has_ones:
    #             print(i, " = YES")
    #         else:
    #             print(i, " = NO")
    #         i += 1
    #     j += 1

    neg_weights = torch.pow(1 - gt, 4)
    
    loss = 0
    for pre in pred: # 预测值
        # 约束在0-1之间
        pre = torch.clamp(torch.sigmoid(pre), min=1e-4, max=1 - 1e-4)
        pos_loss = torch.log(pre) * torch.pow(1 - pre, 2) * pos_inds
        neg_loss = torch.log(1 - pre) * torch.pow(pre,2) * neg_weights * neg_inds
        
        pos_loss[torch.isinf(pos_loss) | torch.isnan(pos_loss)] = 0
        neg_loss[torch.isinf(neg_loss) | torch.isnan(neg_loss)] = 0
        
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss # 只有负样本
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss / len(pred)

    # loss = 0
    
    # # epsilon = 1e-5
    # # pred = torch.clamp(pred, epsilon, 1 - epsilon)
    # # print(torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds)

    # pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    # neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    
    # pos_loss[pos_loss.isnan()] = 0
    # neg_loss[torch.isinf(neg_loss) | torch.isnan(neg_loss)] = 0
    
    # # print(neg_loss)
    # # num_pos  = pos_inds.float().sum()
    # # pos_loss = pos_loss.sum()
    # # neg_loss = neg_loss.sum()
    
    # for i in range(len(pred)):
    #     num_pos = pos_inds[i].float().sum()
    #     # print("pos = ", pos_loss[i].float().sum())
    #     # print("neg = ", neg_loss[i].float().sum())
    #     if num_pos == 0:
    #         loss = loss - neg_loss[i].float().sum()
    #     else:
    #         loss = loss - (pos_loss[i].float().sum() + neg_loss[i].float().sum()) / num_pos
    # return loss/len(pred)


def DiceLoss(inputs, target):
    inputs = torch.nn.Sigmoid()(inputs)
    # inputs = (inputs > 0.5).float()
    if target.sum() == 0:
        if inputs.sum() == 0:
            return torch.tensor(1., device="cuda")
        else:
            return torch.tensor(0., device="cuda")
    intersection = torch.sum(inputs * target)
    smooth = 1
    dice = (2 * intersection + smooth) / ((inputs.sum() + target.sum()) * 1.0  + smooth)
    return 1-dice

class Lossncriterion(nn.Module):
    def __init__(self, mode, per_channel=False):
        super(Lossncriterion, self).__init__()
        self.laplacian = torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], dtype=torch.half).requires_grad_(False).cuda()
        self.converter = nn.Sigmoid()
        self.mode = mode
        self.per_channel = per_channel
        self.weight = [0.33,0.33,0.33]
        
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
        
    def forward(self, inputs, target):
        # wbce, wiou = structure_loss(inputs, target)
        # loss = wbce.mean() + wiou.mean()

        if self.mode == "structure_loss":
            wbce, wiou = structure_loss(inputs, target)
            if self.per_channel:
                wbce_per_ch = torch.mean(wbce, dim=0)
                wiou_per_ch = torch.mean(wiou, dim=0)
                loss = wbce_per_ch[0]*self.weight[0] + wbce_per_ch[1]*self.weight[1] + wbce_per_ch[2]*self.weight[2]
                loss += wiou_per_ch[0]*self.weight[0] + wiou_per_ch[1]*self.weight[1] + wiou_per_ch[2]*self.weight[2]
            else:
                loss = wbce.mean() + wiou.mean()
                
        elif self.mode == "negloss":
            loss = _neg_loss(inputs, target)
        elif self.mode == "dice_loss":
            if self.per_channel:
                loss_class0 = DiceLoss(inputs[0], target[0])
                loss_class1 = DiceLoss(inputs[1], target[1])
                loss_class2 = DiceLoss(inputs[2], target[2])
                loss = loss_class0*self.weight[0] + loss_class1*self.weight[1] + loss_class2*self.weight[2]
            else:
                loss = DiceLoss(inputs, target)

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
