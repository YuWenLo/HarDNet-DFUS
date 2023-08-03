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

import numpy as np
import cv2
import json
from skimage import feature
from pathlib import Path
# -

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=int, default=53, help='backbone version')
    parser.add_argument('--class-num', type=int, default=3, help='output class')
    parser.add_argument('--test-size', type=int, default=1024, help='training dataset size')

    parser.add_argument('--weight', type=str, default='', help='path to model weight')
    parser.add_argument('--modelname', type=str, default='fchardnet', help='choose model')
    parser.add_argument('--tta', action='store_true', help='testing time augmentation')
    parser.add_argument('--save_path', type=str, default='pred_mask', help='path to save mask')
    parser.add_argument('--data_path', nargs='+', type=str, default='../DFUC2022_val' , help='path to testing data')
    parser.add_argument('--weight_mode', type=str, default='final', help='weight')
    parser.add_argument('--min_distance', type=int, default=3, help='find cell')
    
    parser.add_argument('--rect', action='store_true', help='padding the image into rectangle')
    parser.add_argument('--visualize', action='store_true', help='visualize the ground truth and prediction on original image')
    parser.add_argument('--threshold', type=float,default=0.5, help='threshold for mask')
    
    return parser.parse_args()

def test(model, test_data, tta):
    FPS = AvgMeter()
    pbar = enumerate(test_data)
    pbar = tqdm(pbar, total=len(test_data))
    heatmaps = []
    
    for i, (image, name, shape, img) in pbar:
        image = image.cuda()
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            output = model(image)
            if tta:
                output = (output + tf.vflip(tf.hflip(model(tf.vflip(tf.hflip(image))))))/2
                
        output = F.interpolate(output, size=shape, mode='bilinear', align_corners=False)
        
        # output = nn.Tanh()(output).squeeze().cpu().numpy()
        # output = (output - output.min()) / (output.max() - output.min() + 1e-16)
        output = torch.softmax(output, dim=1)
        
        torch.cuda.synchronize()
        elapsed_time = time.perf_counter() - start_time
        FPS.update(1/elapsed_time, 1)
        
        pbar.set_description('FPS: %7.4g'%(FPS.avg))
        
        heatmaps.append(output)
            
    return heatmaps

def find_cells(heatmap, min_distance, threshold):
    """This function detects the cells in the output heatmap

    Parameters
    ----------
    heatmap: torch.tensor
        output heatmap of the model,  shape: [1, 3, 1024, 1024]

    Returns
    -------
        List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
    """
    arr = heatmap[0,:,:,:].cpu().detach().numpy()
    # arr = np.transpose(arr, (1, 2, 0)) # CHW -> HWC

    bg, pred_wo_bg = np.split(arr, (1,), axis=0) # Background and non-background channels
    bg = np.squeeze(bg, axis=0)
    obj = 1.0 - bg

    arr = cv2.GaussianBlur(obj, (0, 0), sigmaX=3)
    peaks = feature.peak_local_max(
        arr, min_distance=min_distance, exclude_border=0, threshold_abs=threshold
    ) # List[y, x]

    maxval = np.max(pred_wo_bg, axis=0)
    maxcls_0 = np.argmax(pred_wo_bg, axis=0)

    # Filter out peaks if background score dominates
    peaks = np.array([peak for peak in peaks if bg[peak[0], peak[1]] < maxval[peak[0], peak[1]]])
    if len(peaks) == 0:
        return []

    # Get score and class of the peaks
    scores = maxval[peaks[:, 0], peaks[:, 1]]
    peak_class = maxcls_0[peaks[:, 0], peaks[:, 1]]

    predicted_cells = [(x, y, c + 1, float(s)) for [y, x], c, s in zip(peaks, peak_class, scores)]

    return predicted_cells

class DetectionWriter:
    def __init__(self, output_path):
        if output_path.suffix != '.json':
            output_path = output_path / '.json' 

        self._output_path = output_path
        self._data = {
            "type": "Multiple points",
            "points": [],
            "version": {"major": 1, "minor": 0},
        } 

    def add_point(
            self, 
            x: int, 
            y: int,
            class_id: int,
            prob: float, 
            sample_id: int
        ):
        """Recording a single point/cell

        Parameters
        ----------
        x: int
            Cell's x-coordinate in the cell patch
        y: int
            Cell's y-coordinate in the cell patch
        class_id: int
            Class identifier of the cell, either 1 (BC) or 2 (TC)
        prob: float
            Confidence score
        sample_id: str
            Identifier of the sample
        """
        point = {
            "name": "image_{}".format(str(sample_id)),
            "point": [int(x), int(y), int(class_id)],
            "probability": prob}
        self._data["points"].append(point)

    def add_points(self, points, sample_id: str):
        """Recording a list of points/cells

        Parameters
        ----------
        points: List
            List of points, each point consisting of (x, y, class_id, prob)
        sample_id: str
            Identifier of the sample
        """
        for x, y, c, prob in points:
            self.add_point(x, y, c, prob, sample_id)

    def save(self):
        """This method exports the predictions in Multiple Point json
        format at the designated path. 
        
        - NOTE: that this will fail if not cells are predicted
        """
        assert len(self._data["points"]) > 0, "No cells were predicted"
        with open(self._output_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=4)
        print(f"Predictions were saved at `{self._output_path}`")


if __name__ == '__main__':
    opt = arg_parser()
    
    test_data = test_dataset(opt.data_path, opt.test_size, opt.rect)
    model = build_model(opt.modelname, opt.class_num, opt.arch)
    print(opt.weight)
    model.load_state_dict(torch.load(opt.weight))#, map_location=device))
    model.eval()
        
    heatmaps = test(model, test_data, opt.tta)

    parent_dir = os.path.dirname(opt.weight)
    file_name = opt.weight_mode + "_fc_prediction_point.json"
    
    output_path_str = os.path.join(parent_dir, file_name)
    output_path = Path(output_path_str)
    writer = DetectionWriter(output_path)
    
    i = 0
    print("now min distance = ", opt.min_distance, ", threshold = ", opt.threshold)
    for heatmap in heatmaps:
        cell_classification = find_cells(heatmap, opt.min_distance, opt.threshold)
        writer.add_points(cell_classification, i)
        i = i + 1 
    writer.save()