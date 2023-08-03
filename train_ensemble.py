# +
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from torch.cuda import amp

import os
import argparse
import math
import random
import time
import logging
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from utils.utils import AvgMeter, build_model
from utils.dataloader import create_dataset
from utils.ema import ModelEMA
from utils.loss import Lossncriterion, structure_loss
from utils.optim import set_optimizer

import torch.nn as nn
# -
def arg_parser():
    parser = argparse.ArgumentParser()
    # for training
    parser.add_argument('--epoch', type=int, default=300, help='# epoch')
    parser.add_argument('--batchsize', type=int, default=16, help='batch size')
    parser.add_argument('--kfold', type=int, default=1, help='# fold')
    parser.add_argument('--k', type=int, default=-1, help='specific # fold')
    parser.add_argument('--seed', type=int, default=42, help='random seed for split data')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--name', type=str, default='exp', help='exp name to annotate this training')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='choose optimizer')
    parser.add_argument('--loss', type=str, default='structure_loss', help='choose optimizer')
    parser.add_argument('--per_channel', action='store_true', help='calculate loss by per channel')
    parser.add_argument('--test', action='store_true', help='run test')
    parser.add_argument('--save_mask_indice', action='store_true', help='save tissue indice')
    parser.add_argument('--color_ver', type=int, default=-1, help='colorjitter_version')

    # for data
    parser.add_argument('--dataratio', type=float,default=0.8, help='ratio of data for training/val')
    parser.add_argument('--data_path', nargs='+', type=str, default='../', help='path to training data')
    parser.add_argument('--augmentation', action='store_true', help='activate data augmentation')
    parser.add_argument('--cell_size', type=int, default=5, help='')

    # for model
    parser.add_argument('--tta', action='store_true', help='testing time augmentation')
    parser.add_argument('--ensemble', action='store_true', help='ensemble or not')
    parser.add_argument('--class_num', type=int, default=3, help='output class')
    parser.add_argument('--arch', type=int, default=53, help='backbone version')
    parser.add_argument('--trainsize', type=int, default=1024, help='img size')
    parser.add_argument('--weight', type=str, default='', help='path to model weight')
    parser.add_argument('--modelname', type=str, default='lawinloss4', help='choose model')
    parser.add_argument('--decoder', type=str, default='lawin', help='choose decoder')
    parser.add_argument('--rect', action='store_true', help='padding the image into rectangle')

    return parser.parse_args()

def trainingplot(rec, name):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
            
    x = np.arange(0, opt.epoch, 1)
    line1, = ax.plot(x, rec[0, :], label='Loss')
    line2, = ax.plot(x, rec[1, :], label='Deep loss 1')
    line3, = ax.plot(x, rec[2, :], label='Deep loss 2')
    line4, = ax2.plot(x, rec[3, :], label='Dice', color='r')
    line5, = ax2.plot(x, rec[4, :], label='Best dice', color='y')

    line6, = ax.plot(x, rec[5, :], label='Val loss')
            
    ax.set_xlabel('Epoch')
    ax.set_title(name, fontsize=16)
            
    ax.set_ylim([0, 2])
    ax2.set_ylim([0, 1])
            
    ax.set_ylabel('Loss', color='g')
    ax2.set_ylabel('Dice', color='b')
            
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.savefig(name, dpi=300)

def test(model, criterion, test_loader, modelname):
    model.eval()
    mdice, mwbce, mwiou, omax, omin = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    pbar = enumerate(test_loader)
    print(('\n' + '%10s' * 6) % ('Dice', 'gpu_mem', 'wbce', 'wiou', 'max', 'min'))
    pbar = tqdm(pbar, total=len(test_loader))

    # mdice0, mdice1, mdice2 = 0,0,0
    
    for i, (image, gt, name) in pbar:
        gt = gt.cuda()
        image = image.cuda()
        with torch.no_grad():
            output = model(image)
            
        wbce, wiou = structure_loss(output, gt)
        dice = criterion.dice_coefficient(output, gt)

        # mdice0 += criterion.dice_coefficient(output[0][0], gt[0][0])
        # mdice1 += criterion.dice_coefficient(output[0][1], gt[0][1])
        # mdice2 += criterion.dice_coefficient(output[0][2], gt[0][2])


        if modelname == 'fchardnet':
            mdice.update(dice, 1)
            mwbce.update(wbce, 1)
            mwiou.update(wiou, 1)
        else:
            mdice.update(dice.item(), 1)
            mwbce.update(wbce.item(), 1)
            mwiou.update(wiou.item(), 1)
        omax.update(output.max().item(), 1)
        omin.update(output.min().item(), 1)
        

        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
        if modelname == 'fchardnet':
            s = ('%10.4g' + '%10s' + '%10.4g' * 2) % (mdice.avg, mem, omax.avg, omin.avg)
        else:
            s = ('%10.4g' + '%10s' + '%10.4g' * 4) % (mdice.avg, mem, mwbce.avg, mwiou.avg, omax.avg, omin.avg)
        pbar.set_description(s)


    # mdice0 = round((mdice0/len(test_loader)).item(),4)
    # mdice1 = round((mdice1/len(test_loader)).item(),4)
    # mdice2 = round((mdice2/len(test_loader)).item(),4)

    # print("dice=(", mdice0, ",", mdice1,",", mdice2, ")")

    if modelname == 'fchardnet':  
        avg = mwbce.avg+mwiou.avg
        return mdice.avg, torch.mean(avg)
    else:
        return mdice.avg, mwbce.avg+mwiou.avg

def save_mask(model, test_loader, fold_result, num_fold=5, k=0, tta=False):
    model.eval()
    FPS = AvgMeter()
    pbar = enumerate(test_loader)
    pbar = tqdm(pbar, total=len(test_loader))
    save_path = "/work/wagw1014/OCELOT/tissue/tta_all_tissue_strloss_650e_prediction_indice/"
    os.makedirs(save_path, exist_ok=True)
    
    heatmaps = []
    for i, (image, gt, name) in pbar:
        gt = gt.cuda()
        image = image.cuda()
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            output = model(image)
            if tta:
                output = (output + tf.vflip(tf.hflip(model(tf.vflip(tf.hflip(image))))))/2
            
        output = F.interpolate(output, size=(1024,1024), mode='bilinear', align_corners=False)
        
        if fold_result != None:
            output += fold_result[i]
        if (num_fold - k) == 1:
            output = output/num_fold
            maxcls_0 = np.argmax(output[0].cpu().detach().numpy(), axis=0)
            np.save(save_path + str(i) + '.npy', maxcls_0)
                
            torch.cuda.synchronize()
            elapsed_time = time.perf_counter() - start_time
            FPS.update(1/elapsed_time, 1)
            pbar.set_description('FPS: %7.4g'%(FPS.avg))
        
        heatmaps.append(output)
    return heatmaps


def train(train_loader, model, optimizer, epoch, opt, scaler, ema, criterion, modelname):
    model.train()
    loss_record, deep1, deep2, boundary = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    pbar = enumerate(train_loader)
    print(('\n' + '%10s' * 6) % ('Epoch', 'gpu_mem', 'loss', 'deep1', 'deep2', 'bound'))
    if opt.global_rank in [-1, 0]:
        pbar = tqdm(pbar, total=len(train_loader))
    
    for i, (images, gts, name) in pbar:
        images = images.cuda()
        gts = gts.cuda()

        # multiscale = 0.25
        # trainsize = random.randrange(int(opt.trainsize * (1 - multiscale)), int(opt.trainsize * (1 + multiscale))) // 64 * 64
        images = F.interpolate(images, size=(opt.trainsize, opt.trainsize), mode='bilinear', align_corners=False)
        gts = F.interpolate(gts, size=(opt.trainsize, opt.trainsize), mode='bilinear', align_corners=False)
        
        # ---- forward ----
        with amp.autocast():
            output = model(images)
            if modelname == 'fchardnet':
                loss = criterion(output, gts)
                scaler.scale(loss).backward()
                loss_record.update(loss.item(), opt.batchsize)
            
            else:
                loss = criterion(output[0], gts)
                deep_loss = criterion(output[1], gts)
                deep_loss2 = criterion(output[2], gts)
                # boundary_loss = criterion.boundary_forward(output[3], gts)
            
                scaler.scale(loss).backward(retain_graph=True)
                scaler.scale(deep_loss).backward(retain_graph = True)
                scaler.scale(deep_loss2).backward()#retain_graph = True)
                # scaler.scale(boundary_loss).backward()
                    
                loss_record.update(loss.item(), opt.batchsize)
                deep1.update(deep_loss.item(), opt.batchsize)
                deep2.update(deep_loss2.item(), opt.batchsize)
                # boundary.update(boundary_loss.item(), opt.batchsize)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        ema.update(model)
        
        if opt.global_rank in [-1, 0]:
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 4) % ('%g/%g' % (epoch, opt.epoch - 1), mem, loss_record.avg, deep1.avg, deep2.avg, boundary.avg)
            pbar.set_description(s)
            ema.update_attr(model)
    
    return loss_record.avg, deep1.avg, deep2.avg



if __name__ == '__main__':
    opt = arg_parser()
    if opt.seed == None:
        opt.seed = np.random.randint(2147483647)
        print('You choosed seed %g in this training.'%opt.seed)
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1   
    
    if opt.kfold == 1:
        save_path = os.path.join('/work/wagw1014/OCELOT/fc/weights_7_19', opt.name)
        os.makedirs(save_path, exist_ok=True)
        
        logname = save_path + '/' + opt.name + '_' + time.strftime("%Y%m%d%H%M%S", time.localtime()) + '.log'
        logging.basicConfig(filename=logname, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
        logging.info(opt)
        print('logging at ', logname)
    
    
    for k in range(opt.kfold):
        if opt.kfold != 1:

            print('%g/%g-fold'%(k+1, opt.kfold))
            if opt.k != -1:
                k = opt.k

            save_path = os.path.join('/work/wagw1014/OCELOT/fc/weights_7_19', opt.name, str(k))
            os.makedirs(save_path, exist_ok=True)
            
            logname = save_path + '/' + opt.name + '_' + time.strftime("%Y%m%d%H%M%S", time.localtime()) + '.log'
            logging.basicConfig(filename=logname, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
            logging.info(opt)
            print('logging at ', logname)

        # if opt.kfold > 1:
        #     print('%g/%g-fold'%(k+1, opt.kfold))
        #     if opt.k != -1:
        #         k = opt.k
                
     
        train_dataset = create_dataset(opt.data_path, opt.trainsize, opt.augmentation, True, opt.dataratio, opt.rect, k=k, k_fold=opt.kfold, seed=opt.seed, num_class=opt.class_num, cell_size=opt.cell_size, color_ver=opt.color_ver)
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=4, pin_memory=True)
        del train_dataset

        test_dataset = create_dataset(opt.data_path, opt.trainsize, False, False, opt.dataratio, opt.rect, k=k, k_fold=opt.kfold, seed=opt.seed, num_class=opt.class_num, cell_size=opt.cell_size)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        del test_dataset
        
        model = build_model(opt.modelname, opt.class_num, opt.arch)
        logging.info(model)
            
        optimizer = set_optimizer(model, opt.optimizer, opt.lr)
        logging.info(optimizer)
        lf = lambda x: ((1.001 + math.cos(x * math.pi / opt.epoch))) #* (1 - 0.1) + 0.1  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        scaler = amp.GradScaler()
        ema = ModelEMA(model) if opt.global_rank in [-1, 0] else None
        if opt.loss == "negloss" or opt.loss == "structure_loss" or opt.loss == "bootstrapped_cross_entropy2d" or opt.loss == "dice_loss":
            print("loss = ", opt.loss, ", per_channel = ", opt.per_channel)
            criterion = Lossncriterion(mode=opt.loss, per_channel=opt.per_channel).cuda()
        elif opt.loss == "ce":
            print("ce")
            criterion = nn.CrossEntropyLoss()
        elif opt.loss == "bce":
            print("bce")
            criterion = nn.BCEWithLogitsLoss()
        elif opt.loss == "mse":
            print("mse")
            criterion = nn.MSELoss(reduction = 'sum')

        
        if opt.weight != '' and not opt.ensemble:
            if opt.modelname == 'fchardnet' and not opt.test and not opt.save_mask_indice:
                print("load base")
                model.base.load_state_dict(torch.load(opt.weight))
            else:
                print("load all")
                model.load_state_dict(torch.load(opt.weight))#, map_location=device))
        
        if opt.test:
            print("Start testing!")
            meandice, val_loss = test(model, criterion, test_loader, opt.modelname)
            print("loss=", round(val_loss.item(),4), ",mdice=", round(meandice.item(),4))
            
        elif opt.save_mask_indice:
            print("Start mask indice!")
            
            weights_dir = sorted(os.listdir(opt.weight))
            fold_result = None
            num_fold = len(weights_dir)
            print("num of fold = ", num_fold)
            for k, weight in enumerate(weights_dir):
                weight = os.path.join(opt.weight, weight)
                print('Test %gth weight'%(k), weight)
                model.load_state_dict(torch.load(weight))
                model.eval()
                fold_result = save_mask(model, test_loader, fold_result, num_fold, k, opt.tta)
            

        else:
            print('Start training at rank: ', opt.global_rank)
            print("save path = ", save_path)
            best = 0
            rec = np.zeros((6, opt.epoch))
            for epoch in range(opt.epoch):
                optimizer.zero_grad()
                loss, deep1, deep2= train(train_loader, model, optimizer, epoch, opt, scaler, ema, criterion, opt.modelname)
                scheduler.step()

                meandice, val_loss = test(model, criterion, test_loader, opt.modelname)
                if meandice > best:
                    best = meandice
                    pthname = os.path.join(save_path, 'epoch%s_%g,%g_best_%g.pth'%(str(epoch), k+1, opt.kfold, int(best*10000)))
                    torch.save(model.state_dict(),  pthname)
                    print('[Saving Best Weight]', pthname)
                rec[0, epoch] = loss
                rec[1, epoch] = deep1
                rec[2, epoch] = deep2
                rec[3, epoch] = meandice
                rec[4, epoch] = best
                rec[5, epoch] = val_loss
                logging.info('Epoch: %g,mDice: %g,Best mDice: %g,loss: %g,loss2: %g,loss3: %g,lr: %g'%(epoch, meandice, best, loss, deep1, deep2, scheduler.get_last_lr()[0]))
                print("best meandice: ", best)

                if epoch+1 >= 100 and (epoch+1)%50==0:
                    pthname = os.path.join(save_path, '%s%g_%g.pth'%("epoch", epoch, int(meandice*10000)))
                    torch.save(model.state_dict(),  pthname)

            torch.save(model.state_dict(), os.path.join(save_path, '%s_%g,%g_final_%g.pth'%(opt.name, k+1, opt.kfold, int(best*10000))))
            trainingplot(rec, os.path.join(save_path, '%s_%g,%g_final_%g.pdf'%(opt.name, k+1, opt.kfold, int(best*10000))))
            
            if opt.k != -1:
                break
