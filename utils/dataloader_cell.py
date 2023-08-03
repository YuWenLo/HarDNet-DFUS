# cell

# albumentations
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import os
import glob
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.utils import square_padding
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split



# +
def calculatemns(img_list, size, rect):
    ''' Calculate mean and std. of images
    Args:
        img_list: list of image name
        size: target size in training
        rect: padding to rect or not
    '''
    mean = 0.
    std = 0.
    for name in img_list:
        image = Image.open(name).convert('RGB')
        w, h = image.size
        if rect:
            image = square_padding(image, w, h)
        
        image = transforms.Resize((size, size))(transforms.ToTensor()(image))
        image = image.flatten(1)
        mean += image.mean(1)
        std += image.std(1)

    mean /= len(img_list)
    std /= len(img_list)
    return mean, std


def split_data(length, ratio, k=0, seed=42, k_fold=1):
    ''' Randomly choose the index of training/dalidation data
    Args:
        length: length of collected data
        ratio: ratio of data for training, not worked if trained w/ k-fold
        k: # fold in k-fold
        seed: seed for reproducing the random result
        k_fold: # fold for cross-validation
    '''
    train_size = int(ratio * length)
    valid_size = length - train_size
    train_indices, val_indices = train_test_split(
        np.linspace(0, length - 1, length).astype("int"),
        test_size=valid_size,
        random_state=seed,
    )
    
    return train_indices, val_indices, val_indices


# -

class create_dataset(data.Dataset):
    ''' collect all files in data_path and determine augmentation
    Args:
        data_path: the path that contains images and masks.
        trainsize: resize all images to trainsize for training
        augmentation: enable data augmentation or not
        train: determine the dataset is for training or validation
        train_ratio: ratio of data for training
        rect: padding image to square before resize to keep its aspect ratio
        k: # fold in k-fold
        k_fold: # fold for cross-validation
        seed: seed for reproducing the random result
    '''
    def __init__(self, data_path, trainsize, augmentations, train=True, train_ratio=0.8, rect=False, k=0, k_fold=1, seed=42, num_class=3, cell_size=5, color_ver=-1):
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.ratio = train_ratio
        self.rect = rect
        self.cell_size = cell_size
        self.color_ver = color_ver
        print("color ver = ", self.color_ver)
        try:
            '''
            We assert that your folder of images/masks is named by "images"/"masks"
            and their type are .jpg or .png
            '''
            f = []
            for p in data_path if isinstance(data_path, list) else [data_path]:
                p = Path(p)
                f += glob.glob(str(p / '**' / '*.*'), recursive=False)

            self.images = sorted([x for x in f if ('images' in x) and (x.endswith('.jpg') or x.endswith('.png'))])
            gt_path = 'heatmap_radius5'
            print(gt_path)
            self.gts = sorted([x for x in f if (gt_path in x) and (x.endswith('.npy'))])
            length = len(self.images)
            
            #mean, std = calculatemns(self.images, self.trainsize, self.rect)
            #print('mean:', mean, ' std:', std)
            if self.ratio != 1.0:
                train_idx, val_idx, test_idx = split_data(length, self.ratio, k=k, seed=seed, k_fold=k_fold)
            
            if train:
                if self.ratio != 1.0:
                    self.images = sorted([self.images[idx] for idx in train_idx])
                    self.gts = sorted([self.gts[idx] for idx in train_idx])
                
                for i in range(len(self.images)):
                    assert self.images[i].split(os.sep)[-1].split('.')[0] == self.gts[i].split(os.sep)[-1].split('.')[0]
                print('load %g training data from %g images in %s'%(len(self.images), length, data_path))
            
            else:
                if self.ratio != 0.0:
                    self.images = sorted([self.images[idx] for idx in val_idx])
                    self.gts = sorted([self.gts[idx] for idx in val_idx])
                
                for i in range(len(self.images)):
                    assert self.images[i].split(os.sep)[-1].split('.')[0] == self.gts[i].split(os.sep)[-1].split('.')[0]
                print('load %g validation data from %g images in %s'%(len(self.images), length, data_path))
        
        except Exception as e:
            raise Exception('Error loading data from %s: %s\n' % (data_path, e))
        
        self.size = len(self.images)
        if self.augmentations == True:
            print("data augmentation 2")
            self.transform = A.Compose([
                # A.OneOf([
                #     A.CenterCrop(self.trainsize, self.trainsize, p=1),
                #     A.RandomCrop(self.trainsize, self.trainsize, p=1),
                # ], p=0.3),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1 ,scale_limit=0.1, rotate_limit=45, p=0.5),
                A.OneOf([
                    A.CoarseDropout(max_holes=8, max_height=20, max_width=20, min_holes=None, min_height=None, min_width=None, fill_value=0, p=1),
                    A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=1),
                ], p=0.5)
            ],additional_targets={'mask1': 'mask', 'mask2': 'mask'})
   
        else:
            print("no data augmentation")
            self.transform = A.Compose([A.Resize(self.trainsize, self.trainsize)],additional_targets={'mask1': 'mask', 'mask2': 'mask'})
        self.nom = transforms.Normalize([0.7610, 0.5776, 0.6962], [0.1515, 0.1870, 0.1426])
        self.totensor = A.Compose([ToTensorV2()],additional_targets={'mask1': 'mask', 'mask2': 'mask'})
        
        print("cell size = ", self.cell_size)
        
    def __getitem__(self, index):
        # https://github.com/pytorch/vision/issues/9
        seed = np.random.randint(2147483647) # make a seed with numpy generator  #21474
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7

        # Read an image with OpenCV
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        name = self.gts[index].split('/')[-1].split('.')[0]
        name = "/work/wagw1014/OCELOT/radius5_mask_1class/" + name
        # name = "/work/wagw1014/OCELOT/radius{}_mask_1class/".format(self.cell_size) + name
        name_0 = name + '_0.jpg'
        name_1 = name + '_1.jpg'
        name_2 = name + '_2.jpg'
        gt0 = cv2.imread(name_0, cv2.IMREAD_GRAYSCALE)
        gt1 = cv2.imread(name_1, cv2.IMREAD_GRAYSCALE)
        gt2 = cv2.imread(name_2, cv2.IMREAD_GRAYSCALE)
        # gt = np.stack([gt0, gt1, gt2], axis=0)
        # gt = cv2.imread(self.gts[index], cv2.IMREAD_GRAYSCALE)
        # name = self.gts[index]
        
        
        if self.augmentations:
            total = self.transform(image=image, mask=gt0, mask1=gt1, mask2=gt2)
            image = total["image"]
            gt0 = total['mask']
            gt1 = total['mask1']
            gt2 = total['mask2']
            
            # if self.color_ver == 1:
            #     total = A.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0, p=0.3)(image=image)
            # elif self.color_ver == 2:
            #     total = A.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0, p=0.3)(image=image)
            # elif self.color_ver == 3:
            #     total = A.ColorJitter(brightness=0, contrast=0, saturation=0.5, hue=0, p=0.3)(image=image)
            # elif self.color_ver == 4:
            #     total = A.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5, p=0.3)(image=image)
            # image = total["image"]
            
        # total = A.Resize(self.trainsize, self.trainsize)(image=image, mask=gt)
        # image = total["image"]
        # gt = total['mask']
        
        image_final = self.totensor(image=image)
        image = image_final["image"]
        image = self.nom(image)

        gt_final = self.totensor(image=gt0, mask=gt0, mask1=gt1, mask2=gt2)
        gt0 = gt_final['mask']
        gt1 = gt_final['mask1']
        gt2 = gt_final['mask2']

        gt = torch.cat((gt0.unsqueeze(0), gt1.unsqueeze(0), gt2.unsqueeze(0)), 0)
        del gt0, gt1, gt2
        # gt = np.stack([gt0, gt1, gt2], axis=-1)

        # return image, gt.unsqueeze(0), name
        return image, gt, name

    def __len__(self):
        return self.size


class test_dataset(data.Dataset):
    ''' collect all files in data_path and determine augmentation
    Args:
        data_path: the path that contains images and masks.
        size: resize all images to trainsize for training
        rect: padding image to square before resize to keep its aspect ratio
    '''
    def __init__(self, data_path, size, rect):
        self.trainsize = size
        try:
            f = []
            for p in data_path if isinstance(data_path, list) else [data_path]:
                p = Path(p)
                f += glob.glob(str(p / '*.*'), recursive=False)
            self.images = sorted([x for x in f])
            length = len(self.images)
            
        except Exception as e:
            raise Exception('Error loading data from %s: %s\n' % (data_path, e))

        print('load %g all images'%length, 'from', data_path)
        #mean, std = calculatemns(self.images, self.trainsize, rect)
        #print('mean:', mean, ' std:', std)
        
        self.rect = rect
        self.size = len(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(), 
            #transforms.Normalize(mean, std)])
            transforms.Normalize([0.7610, 0.5776, 0.6962], [0.1515, 0.1870, 0.1426])])
            
    def __getitem__(self, index):
        name = self.images[index]
        image = Image.open(name).convert('RGB')
        image0 = np.array(image)
        w, h = image.size

        if self.rect:
            image = square_padding(image, w, h)
        
        image = self.transform(image)
        
        return image.unsqueeze(0), name, (h, w), image0

    def __len__(self):
        return self.size
