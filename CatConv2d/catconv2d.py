import math
from torch import nn
from torch.autograd import Function
import torch

import catconv2d_cuda

torch.manual_seed(42)


class CatConvFunction(Function):
    @staticmethod
    def forward(ctx, input_list, weights, bias, relu):
        outputs = catconv2d_cuda.forward(input_list, weights, bias, relu)
        variables = [weights, bias]
        ctx.save_for_backward(*variables)
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        outputs = catconv2d_cuda.backward(
            grad_output.contiguous(), *ctx.saved_tensors)
        d_input, d_weights, d_bias = outputs
        return d_input, d_weights, d_bias


class CatConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, relu=False, padding_mode='zeros'):
        super(CatConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        #self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *kernel_size) )
        
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.relu = relu
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_channels)
        self.weight.data.uniform_(-stdv, +stdv)
        self.bias.data.fill_(0)

    def forward(self, input_list):
        #print(len(input_list), input_list[0].shape, self.weights.shape, self.bias.shape)
        return CatConvFunction.apply(input_list, self.weight, self.bias, self.relu)
