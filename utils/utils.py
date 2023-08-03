import torch
import cv2
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from lib.HarDMSEG import KingMSEG_lawin_loss, KingMSEG_lawin_loss4
from lib.fchardnet import hardnet

# +
def square_padding(image, w, h):
    ''' padding an image to square, if width is larger
    then pad half of their difference
    +1//2 to avoid odd difference

    PIL/Tensor -> PIL/Tensor
    '''
    dif = w - h
    if dif > 0:
        #  left, top, right and bottom
        image = transforms.Pad((0, (dif+1)//2, 0, dif//2), fill=0, padding_mode='constant')(image)
    else:
        image = transforms.Pad(((abs(dif)+1)//2, 0, abs(dif)//2, 0), fill=0, padding_mode='constant')(image)

    return image

def square_unpadding(image, w, h):
    ''' crop origin part of padded image
    +1//2 to avoid odd difference

    PIL/Tensor -> PIL/Tensor
    '''
    dif = w - h
    if dif > 0:
        #  left, top, right and bottom
        image = image[:, :, (dif + 1)//2: (dif + 1)//2 + h, :]
    else:
        image = image[:, :, :, (abs(dif) + 1)//2: (abs(dif) + 1)//2 + w]
    
    
    return image

class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def build_model(modelname='lawinloss4', class_num=1, arch=53):
    print('model:', modelname)
    if modelname == 'lawinloss':
        model = KingMSEG_lawin_loss(class_num=class_num).cuda()
    elif modelname == 'lawinloss4':
        model = KingMSEG_lawin_loss4(class_num=class_num).cuda()
    elif modelname == 'fchardnet':
        model = hardnet(n_classes=class_num).cuda()

    return model

class confusion_matrix():
    """ confusion matrix
    Args:
        inputs (tensor): value in {0, 1}
        target (tensor): value in {0, 1}
    """
    def __init__(self, inputs, target):
        self.dice = AvgMeter()
        self.iou = AvgMeter()
        self.acc = AvgMeter()
        self.precision = AvgMeter()
        self.recall = AvgMeter()
        
    def update(self, inputs, target):
        tpfp = inputs.sum().item()
        tnfn = (inputs == 0).sum().item()
        tpfn = target.sum().item()

        if tpfn == 0 or tpfp == 0:
            if tpfp == 0 and tpfn == 0:
                self.dice.update(1, 1)
                self.iou.update(1, 1)
                self.acc.update(1, 1)
                self.precision.update(1, 1)
                self.recall.update(1, 1)
            else:
                self.dice.update(0, 1)
                self.iou.update(0, 1)
                self.acc.update(0, 1)
                self.precision.update(0, 1)
                self.recall.update(0, 1)
        else:
            tp = (target * inputs).sum().item()
            fp = tpfp - tp
            fn = tpfn - tp
            tn = tnfn - fn
            
            self.dice.update(2*tp / (2*tp+fn+fp), 1)
            self.iou.update(tp / (tp+fn+fp), 1)
            self.acc.update((tp+tn) / (tp+tn+fp+fn), 1)
            self.precision.update(tp / (tp+fp), 1)
            self.recall.update(tp / (tp+fn), 1)
            
            
def save_mask(output, save_path, name, threshold):
    mask = ((output > threshold)*255).astype(np.uint8)
    # fill the holes
    th, im_th = cv2.threshold(mask, 220, 255, cv2.THRESH_BINARY)
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    mask = im_th | im_floodfill_inv
    
    mask = Image.fromarray(mask)
    mask.save(os.path.join(save_path, name))

def visualize_mask(output, save_path, name, threshold, img):
    mask = ((output > threshold)*255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask[..., 0] = 0
    mask[..., 2] = 0
    img = cv2.addWeighted(img, 1, mask, 0.4, 0)
    cv2.imwrite(os.path.join(save_path, name), img)
