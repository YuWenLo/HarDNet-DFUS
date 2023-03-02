# +
import torch
import torchvision.transforms.functional as tf
import torch.nn.functional as F
import torch.nn as nn

import os
import argparse
import time

from tqdm import tqdm
from utils.utils import AvgMeter, square_unpadding, build_model, save_mask, visualize_mask
from utils.dataloader import test_dataset
# -

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=int, default=53, help='backbone version')
    parser.add_argument('--class-num', type=int, default=1, help='output class')
    parser.add_argument('--test-size', type=int, default=384, help='training dataset size')

    parser.add_argument('--weight', nargs='+', type=str, default='weights/lawinloss4', help='path to model weight')
    parser.add_argument('--modelname', type=str, default='lawinloss4', help='choose model')
    parser.add_argument('--tta', type=str, default='', help='testing time augmentation')
    parser.add_argument('--save_path', type=str, default='pred_mask', help='path to save mask')
    parser.add_argument('--data_path', nargs='+', type=str, default='../DFUC2022_val' , help='path to testing data')
    
    parser.add_argument('--rect', action='store_true', help='padding the image into rectangle')
    parser.add_argument('--visualize', action='store_true', help='visualize the ground truth and prediction on original image')
    parser.add_argument('--threshold', type=float,default=0.5, help='threshold for mask')
    
    return parser.parse_args()

def test(model, test_data, fold_result, tta, rect, fold, k, visualize, save_path, threshold):
    FPS = AvgMeter()
    pbar = enumerate(test_data)
    pbar = tqdm(pbar, total=len(test_data))
    result = []
    vis_path = os.path.join(opt.save_path, 'vis')
    
    for i, (image, name, shape, img) in pbar:
        image = image.cuda()
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            output = model(image)
            if tta == 'v':
                output = (output + tf.vflip(model(tf.vflip(image))))/2
            if tta == 'h':
                output = (output + tf.hflip(model(tf.hflip(image))))/2
            if tta == 'vh':
                output = (output + tf.vflip(tf.hflip(model(tf.vflip(tf.hflip(image))))))/2
                
        if rect:
            output = F.interpolate(output, size=(max(shape), max(shape)), mode='bilinear', align_corners=False)
            output = square_unpadding(output, shape[1], shape[0])
        else:
            output = F.interpolate(output, size=shape, mode='bilinear', align_corners=False)
        
        output = nn.Tanh()(output).squeeze().cpu().numpy()
        output = (output - output.min()) / (output.max() - output.min() + 1e-16)
        
        torch.cuda.synchronize()
        elapsed_time = time.perf_counter() - start_time
        FPS.update(1/elapsed_time, 1)
        
        if fold_result != None:
            output += fold_result[i]
        
        # if it's the last turn, process output
        if (fold - k) == 1:
            output = output/fold
            save_mask(output, save_path, name.split(os.sep)[-1].split('.')[0] + '.png', threshold)
            if visualize:
                visualize_mask(output, vis_path, name.split(os.sep)[-1], threshold, img)
        
        result.append(output)
            
        pbar.set_description('FPS: %7.4g, threshold: %7.4g, %g-fold: %gth'%(FPS.avg, threshold, fold, k+1))
            
    return result


if __name__ == '__main__':
    opt = arg_parser()
    os.makedirs(opt.save_path, exist_ok=True)
    if opt.visualize:
        os.makedirs(os.path.join(opt.save_path, 'vis'), exist_ok=True)
    
    test_data = test_dataset(opt.data_path, opt.test_size, opt.rect)
    model = build_model(opt.modelname, opt.class_num, opt.arch)
    
    weightlist = []
    for weight in opt.weight if isinstance(opt.weight, list) else [opt.weight]:
        if os.path.isdir(weight):
            for w in sorted(os.listdir(weight)):
                if '.pth' in w:
                    weightlist.append(os.path.join(weight, w))
            opt.weight = weightlist
    
    fold_result = None
    fold = len(opt.weight)
    for k, weight in enumerate(opt.weight):
        print('Test %gth weight'%(k+1), weight)
        model.load_state_dict(torch.load(weight))
        model.eval()
        fold_result = test(model, test_data, fold_result, opt.tta, opt.rect, fold, k, opt.visualize, opt.save_path, opt.threshold)
