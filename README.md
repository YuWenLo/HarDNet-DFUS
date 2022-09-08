# HarDNet-DFUS: Enhancing Backbone and Decoder of HarDNet-MSEG for Diabetic Foot Ulcer Image Segmentation and Colonoscopy Polyp Segmentation


## 1st Place Winner in MICCAI DFUC 2022!
Official PyTorch implementation of HarDNet-DFUS, contains the prediction codes for our submission to the **Diabetic Foot Ulcer Segmentation Challenge 2022 (DFUC2022)** at **MICCAI2022**.


## HarDNet Family
*inference on V100*
#### For Image Classification : [HarDNet](https://github.com/PingoLH/Pytorch-HarDNet) 78.0 top-1 acc. / 1029.76 Throughput on ImageNet-1K @224x224
#### For Object Detection : [CenterNet-HarDNet](https://github.com/PingoLH/CenterNet-HarDNet) 44.3 mAP / 60 FPS on COCO val @512x512
#### For Semantic Segmentation : [FC-HarDNet](https://github.com/PingoLH/FCHarDNet)  75.9% mIoU / 90.24 FPS on Cityscapes test @1024x2048
#### For Polyp Segmentation : [HarDNet-MSEG](https://github.com/james128333/HarDNet-MSEG) 90.4% mDice / 119 FPS on Kvasir-SEG @352x352


## Main Results
### Performance on DFUC2022 Challenge Dataset
We improve HarDNet-MSEG, enhancing its backbone and decoder for DFUC.

| Method | DFUC Val. Stage <br> mDice | DFUC Val. Stage <br> mIoU | DFUC Testing Stage <br> mDice | DFUC Testing Stage <br>  mIoU |
| :---: | :---: | :---: | :---: | :---: |
| HarDNet-MSEG  | 65.53 | 55.22 | n/a | n/a |
| **HarDNet-DFUS**  |  **70.63**  | **60.49** | **72.87** | **62.52** |

### Performance on Polyp Segmentation
| Method | Kvasir <br> mDice | ClinicDB <br> mIoU | ColonDB <br> mDice | ETIS <br>  mIoU | CVC-T <br>  mIoU | FPS |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| HarDNet-MSEG  | 0.912 | 0.932 | 0.731 | 0.677 | 0.887 | **108** |
| HarDNet-DFUS  |  0.919  | **0.939** | **0.774** | **0.739** | 0.880 | 30 |
| HarDNet-DFUS 5-Fold  |  **0.924**  | 0.932 | 0.773 | 0.730 | **0.896** | 6 |

### Sample Inference and Visualized Results of [FUSeg Challenge Dataset](https://github.com/uwm-bigdata/wound-segmentation/tree/master/data/Foot%20Ulcer%20Segmentation%20Challenge)

<p align="center">
<img src="SAMPLE.png" width=90% height=90% 
class="center">
</p>

## HarDNet-DFUS Architecture
<p align="center">
<img src="DFUS.png" width=190% height=100% 
class="center">
</p>


## Installation & Usage
### Environment setting (Prerequisites)
```
conda create -n dfuc python=3.6
conda activate dfuc
pip install -r requirements.txt
```

### Training

1. Download [weights](https://drive.google.com/drive/folders/1UbuMKLUlCsZAusUVLJqwcBaXiwe0ZUe8?usp=sharing) and place in the folder ``` /weights ``` 
2. Run:
    ```
    python train.py --rect --augmentation --data_path /path/to/training/data

    Optional Args:
    --rect         Padding image to square before resize to keep its aspect ratio
    --augmentation Activating data audmentation during training
    --kfold        Specifying the number of K-Fold Cross-Validation
    --k            Training the specific fold of K-Fold Cross-Validation
    --dataratio    Specifying the ratio of data for training
    --seed         Reproducing the result of data spliting in dataloader
    --data_path    Path to training data
    ```

### Testing

Run:
```
python test.py --rect --weight path/to/weight/or/folder --data_path path/to/testing/data

Optional Args:
--rect         Padding image to square before resize to keep its aspect ratio
--tta          Test time augmentation, 'v/h/vh' for verti/horiz/verti&horiz flip
--weight       It can be a weight or a fold. If it's a folder, the result is the mean of each weight result
--data_path    Path to testing data
--save_path    Path to save prediction mask
```

### Evaluation
Run:
```
python train.py --eval --rect --weight path/to/weight/or/folder --data_path path/to/testing/data

Optional Args:
--rect         Padding image to square before resize to keep its aspect ratio
--weight       It can be a weight or a fold. If it's a folder, the result is the mean of each weight result
--data_path    Path to evaluated data
```

## Reproduce our best submission in DFUC 2022 Challenge Testing Stage 

1. Download [the weights for HarDNet-DFUS](https://drive.google.com/drive/folders/15hhsl1CIvOqa60friINmhnMB3qKRD-5p?usp=sharing) and place them in the same folder, specifying the folder in --weight when testing. (Please ensure there is no other weight in the folder to obtain the same result.)   
2. Run **HarDNet-DFUS with 5-fold cross validation and TTA vhflip** : 
    ```
    python test.py --rect --modelname lawinloss4 --weight /path/to/HarDNet-DFUS_weight/folder --data_path /path/to/testing/data --tta vh
    ```

## Acknowledgement
- This research is supported in part by a grant from the **Ministry of Science and Technology (MOST) of Taiwan**.   
We thank **National Center for High-performance Computing (NCHC)** for providing computational and storage resources.        
