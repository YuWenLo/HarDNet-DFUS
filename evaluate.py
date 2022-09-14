import os
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse

class test_dataset:
    def __init__(self, image_root, gt_root):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.ToTensor()
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_root', type=str, default='/home/u8203550/Kvasir/predict/')
    parser.add_argument('--gt_root', type=str, default='/home/u8203550/Kvasir/mask/')
    opt = parser.parse_args()

    ###########################################################
    ##### image_root : your model inference results' path   ###
    ##### gt_root : gt files' path                          ###
    ###########################################################

    image_root = opt.image_root
    gt_root = opt.gt_root

    test_loader = test_dataset(image_root, gt_root)
    b=0.0

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image
        input = image[0,1,:,:]
        input = np.array(input)
        
        target = np.array(gt)
        N = gt.shape
        smooth = 1
    
        input_flat = np.reshape(input,(-1))
        target_flat = np.reshape(target,(-1))
        
        intersection = (input_flat*target_flat)

        loss =  (2 * intersection.sum() + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
        #loss =  loss.sum() / N[1]
        a =  '{:.4f}'.format(loss)
        a = float(a)
        
        b = b + a
        #print( i, a)

    print('mean dice = ', b/test_loader.size)