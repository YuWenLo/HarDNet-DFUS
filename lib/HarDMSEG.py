#!/usr/bin/env python
# coding: utf-8
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from .kingnet import kingnet53
from .lawinloss import LawinHead5, LawinHead2


class KingMSEG_lawin_loss(nn.Module):
    def __init__(self, class_num=1):
        super(KingMSEG_lawin_loss, self).__init__()
        self.backbone = kingnet53(arch=53, depth_wise=False, pretrained = True)
        self.head = LawinHead2(in_channels=[140, 540, 800, 1200], num_classes = class_num)
        self.last3_seg = nn.Conv2d(512, class_num, kernel_size=1)
    
    def forward(self, x):
        kingnetout = self.backbone(x)
        x_4 = kingnetout[0]
        x_8 = kingnetout[1]
        x_16 = kingnetout[2]
        x_32 = kingnetout[3]

        output, last3_feat, low_level_feat = self.head(x_4, x_8, x_16, x_32)
        output = F.interpolate(output, size=x.size()[2:], mode='bilinear')    
        
        if self.training:
            last3_feat = F.interpolate(self.last3_seg(last3_feat), size=x.size()[2:], mode='bilinear')
            low_level_feat = F.interpolate(low_level_feat, size=x.size()[2:], mode='bilinear')
            return output, last3_feat, low_level_feat

        return output

# w/ deep1, deep2 and boundary
class KingMSEG_lawin_loss4(nn.Module):
    def __init__(self, class_num=1):
        super(KingMSEG_lawin_loss4, self).__init__()
        self.backbone = kingnet53(arch=53, depth_wise=False, pretrained = True)
            
        outch = self.backbone(torch.zeros(1, 3, 512, 512))[-4:]
        outch = [x.size(1) for x in outch]
        self.head = LawinHead5(in_channels=outch, num_classes=class_num)
        self.last3_seg = nn.Conv2d(512, class_num, kernel_size=1)
        self.last3_seg2 = nn.Conv2d(768, class_num, kernel_size=1)
    
    def forward(self, x):
        x_4, x_8, x_16, x_32 = self.backbone(x)[-4:]
        
        output, last3_feat, last3_feat2, low_level_feat = self.head(x_4, x_8, x_16, x_32)
        output = F.interpolate(output, size=x.size()[2:], mode='bilinear')    
        
        if self.training:
            last3_feat = F.interpolate(self.last3_seg(last3_feat), size=x.size()[2:], mode='bilinear')
            last3_feat2 = F.interpolate(self.last3_seg2(last3_feat2), size=x.size()[2:], mode='bilinear')
            low_level_feat = F.interpolate(low_level_feat, size=x.size()[2:], mode='bilinear')

            return output, last3_feat, last3_feat2, low_level_feat

        return output
