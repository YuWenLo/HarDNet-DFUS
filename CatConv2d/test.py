from __future__ import division
from __future__ import print_function

import argparse
import math
import time

import torch
from torch import nn

TIME_SCALES = {'s': 1, 'ms': 1000, 'us': 1000000}

parser = argparse.ArgumentParser()
parser.add_argument('example', choices=['py', 'cpp', 'cuda'])
parser.add_argument('-b', '--batch-size', type=int, default=1)
parser.add_argument('-f', '--features', type=int, default=32)
parser.add_argument('-s', '--state-size', type=int, default=128)
parser.add_argument('-r', '--runs', type=int, default=100)
parser.add_argument('--scale', choices=['s', 'ms', 'us'], default='us')
parser.add_argument('-c', '--cuda', action='store_true')
parser.add_argument('-d', '--double', action='store_true')
options = parser.parse_args()

from catconv2d import CatConv2d
options.cuda = True

device = torch.device("cuda") if options.cuda else torch.device("cpu")
dtype = torch.float64 if options.double else torch.float32

kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': True}


x_list = [15,5,34, 42,7, 9,88,233]

all_list = [
        [24, 96],
        [24, 40, 96],
        [24, 70],
        [24, 40, 70, 96],

        [24, 192],# 40
        [24, 40, 192],# 70
        [24, 70],# 40
        [24, 40, 70, 192],# 118
        [24, 118],# 40
        [24, 40, 118],# 70
        [24, 40, 70, 118, 192],# 200

        [28, 256],# 48
        [28, 48, 256],# 80
        [28, 80],# 48
        [28, 48, 80, 256],# 138
        [28, 138],# 48
        [28, 48, 138],# 80
        [28, 48, 80, 138, 256],# 234

        [36, 320],# 62
        [36, 62, 320],# 104
        [36, 104],# 62
        [36, 62, 104, 320],# 176
        [36, 176],# 62
        [36, 62, 176],# 104
        [36, 62, 104, 176, 320],# 300

        [48, 480],# 82
        [48, 82, 480],# 138
        [48, 138],# 82
        [48, 82, 138, 480],# 236
        [48, 236],# 82
        [48, 82, 236],# 138
        [48, 82, 138, 236, 480],# 400

        [24,24,24,24,118],
        [24,24,24,24,24,24,24,24,200],
        [28,28,28,28,28,28,28,28,234],
        [36,36,36,36,36,36,36,36,300],
        [48,48,48,48,48,48,48,48,400] 
        ]

sz_list  = [128]*4 + [64]*14 +  [32]*14 + [128,64,64,32,32]
ksz_list = [3]*32 + [1]*5
och_list = [
            40, 70, 40, 118, 
            40, 70, 40, 118, 40, 70, 200,
            48, 80, 48, 138, 48, 80, 234,
            62,104, 62, 176, 62,104, 300,
            82,138, 82, 236, 82,138, 400,
            192,256,320,480,720]

img_sz = 87
output_ch = 123

X_list = [torch.randn(options.batch_size, x, img_sz, img_sz).to(device, dtype) for x in x_list]


repeat = int(5*10**12 / (X_list[0].size(2)*X_list[0].size(3)*sum(x_list)*output_ch))

print("in channels = ", sum(x_list))

W = torch.randn(output_ch, sum(x_list), 3, 3).to(device, dtype)
bias = torch.randn(output_ch).to(device, dtype)
relu = nn.ReLU(True)

cconv = CatConv2d( sum(x_list), output_ch, (3,3), relu=False).to(device, dtype)
cconv.weight[:,:,:,:] = W[:,:,:,:]
cconv.bias[:] = bias[:]


conv = nn.Conv2d(sum(x_list), output_ch, kernel_size=3, stride=1, bias=True, padding=1).to(device, dtype)
conv.weight[:,:,:,:] = W[:,:,:,:]
conv.bias[:] = bias[:]

out = cconv(X_list)
out2 = conv(torch.cat(X_list,1))
torch.cuda.synchronize()

print("top 100 difference:", torch.topk((abs(out-out2)).flatten(), 100)[0])



avgs0 = []
avgs1 = []

for i in range(len(all_list)):
  x_list = all_list[i]
  img_sz = sz_list[i]
  output_ch = och_list[i]
  k_sz = ksz_list[i]

  X_list = [torch.randn(options.batch_size, x, img_sz, img_sz).to(device, dtype) for x in x_list]

  repeat = int(1*10**11 / (X_list[0].size(2)*X_list[0].size(3)*sum(x_list)*output_ch*options.batch_size))
  #if i==0:
  #repeat =10
  
  W = torch.randn(output_ch, sum(x_list), k_sz, k_sz).to(device, dtype)
  bias = torch.randn(output_ch).to(device, dtype)

  cconv = CatConv2d( sum(x_list), output_ch, (k_sz,k_sz),relu=False).to(device, dtype)
  cconv.weight[:,:,:,:] = W[:,:,:,:]
  #cconv.bias[:] = bias[:]

  conv = nn.Conv2d(sum(x_list), output_ch, kernel_size=k_sz, stride=1, bias=False, padding=k_sz//2).to(device, dtype)
  conv.weight[:,:,:,:] = W[:,:,:,:]
  #conv.bias[:] = bias[:]
  
  out = cconv(X_list)
  out2 = conv.forward(torch.cat(X_list, 1))
  err = (abs(out-out2) /abs(out2)).mean().data.cpu().numpy()
  if (err > 0.001):
      print(" large error:", err)
      exit()

  out = cconv(X_list)
  out2 = conv.forward(torch.cat(X_list, 1))
  print("in channels = ", sum(x_list), "  out channels = ",output_ch,  " img Size=", img_sz, "  kernel=",k_sz, "  repeat ",repeat, "  error= {0:0.4f}".format(err*100), "%")
  
  torch.cuda.synchronize()


  
  t = time.time()
  for _ in range(repeat):
    out = cconv(X_list)
    #X_list[0] = out
  torch.cuda.synchronize()
  print("CatConv:   ", time.time()-t)
  avgs0 += [(time.time()-t) / repeat * 1000]

  
  out = conv(torch.cat(X_list, 1))
  torch.cuda.synchronize()
  
  t = time.time()
  for _ in range(repeat):
    x = torch.cat(X_list, 1)
    x = conv.forward(x)
  torch.cuda.synchronize()
  
  print("Torch Conv:",time.time()-t)
  avgs1 += [(time.time()-t) / repeat * 1000]

#print(avgs0, avgs1)
improve = (sum(avgs1) - sum(avgs0)) / sum(avgs1)

print(" Torch Conv: ",sum(avgs1),"ms")
print("    CatConv: ",sum(avgs0),"ms", " lantency reduction:", improve*100,"%")

