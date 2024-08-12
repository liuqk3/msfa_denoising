from os import W_OK
import torch
from models.CFNet import *
from fvcore.nn import FlopCountAnalysis, flop_count_table
from thop import profile
from models.CFNet import FreqFilter



'''
FLOPs:
conv_block: (18 * C_out + 20 * C_in + 3) * H_out * W_out * C_out
down: 32 * C_in * C_out * H_out * W_out
up: C_out * C_in * 4 * H_in * W_in + C_out * H_out * W_out

FreqFilter: 
(18 * H_out * W_out - 1) * C_in * C_in * 9 / 8     conv
(C_in * 27 / 8 - 1) * H_out * W_out               softmax
H_out * W_out * C_in * 9 / 8                      invert
2 * C_in * 9 * H_out * W_out - 1                   matrix multiply

HIN: 4 * (C_in //2) * H * W


Params:
conv_block: (10 * C_in + 9 * C_out + 3) * C_out
down: (16 * C_in + 1) * C_out
up: (4 * C_in + 1) * C_out
FreqFilter: 81 * C_in * C_in / 8
HPFB: (3 * C_in + 18 * C_out + 5 + 81/8*C_out) * C_out
'''


def conv_block_flops(C_in, C_out, H_out, W_out):
    return (18 * C_out + 20 * C_in + 3) * H_out * W_out * C_out

def down_flops(C_in, C_out, H_out, W_out):
    return 32 * C_in * C_out * H_out * W_out

def up_flops(C_in, C_out, H_in, H_out, W_in, W_out):
    return C_out * C_in * 4 * H_in * W_in + C_out * H_out * W_out

def FreqFilter_flops(C_in, H_out, W_out):
    return (18 * H_out * W_out - 1) * C_in * C_in * 9 / 8 + (C_in * 27 / 8 - 1) * H_out * W_out + H_out * W_out * C_in * 9 / 8 + 2 * C_in * 9 * H_out * W_out - 1 + C_in * H_out * W_out * 4

def HIN_flops(C_in, H, W):
    return 4 * (C_in //2) * H * W

def HPFB_flops(C_in, C_out, H_out, W_out):
    count = 2 * C_in * C_out * H_out * W_out
    count += C_out * C_out * H_out * W_out * 9 * 2
    count += FreqFilter_flops(C_out, H_out, W_out)
    count += C_in * C_out * H_out * W_out * 2
    count += C_out * H_out * W_out
    count += HIN_flops(C_out, H_out, W_out)
    count += C_out * H_out * W_out
    count += C_out * C_out * H_out * W_out * 9 * 2
    count += C_out * H_out * W_out
    count += C_in * C_out * H_out * W_out * 2
    count += C_out * H_out * W_out

    return count



def conv_block_params(C_in, C_out):
    return (10 * C_in + 9 * C_out + 3) * C_out

def down_params(C_in, C_out):
    return (16 * C_in + 1) * C_out

def up_params(C_in, C_out):
    return (4 * C_in + 1) * C_out

def FreqFilter_params(C_in):
    return 81 * C_in * C_in / 8

def HPFB_params(C_in, C_out):
    return (3 * C_in + 18 * C_out + 5 + 81/8*C_out) * C_out



def count_UNet():
    n1 = 64
    filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
    count_flops = 0
    count_params = 0

    #conv1
    count_flops += conv_block_flops(25, 64, 128, 128)
    count_params += conv_block_params(25, 64)

    count_flops += down_flops(64, 64, 64, 64)
    count_params += down_params(64, 64)

    #conv2
    count_flops += conv_block_flops(64, 128, 64, 64)
    count_params += conv_block_params(64, 128)

    count_flops += down_flops(128, 128, 32, 32)
    count_params += down_params(128, 128)

    #conv3
    count_flops += conv_block_flops(128, 256, 32, 32)
    count_params += conv_block_params(128, 256)

    count_flops += down_flops(256, 256, 16, 16)
    count_params += down_params(256,256)

    #conv4
    count_flops += conv_block_flops(256, 512, 16, 16)
    count_params += conv_block_params(256, 512)

    count_flops += down_flops(512, 512, 8, 8)
    count_params += down_params(512, 512)

    #conv5
    count_flops += conv_block_flops(512, 1024, 8, 8)
    count_params += conv_block_params(512, 1024)

    #Up5
    count_flops += up_flops(1024, 512, 8, 16, 8, 16)
    count_params += up_params(1024, 512)

    #up conv5
    count_flops += conv_block_flops(1024, 512, 16, 16)
    count_params += conv_block_params(1024, 512)

    #Up4
    count_flops += up_flops(512, 256, 16, 32, 16, 32)
    count_params += up_params(512, 256)

    #up conv4
    count_flops += conv_block_flops(512, 256, 32, 32)
    count_params += conv_block_params(512, 256)

    count_flops += up_flops(256, 128, 32, 64, 32, 64)
    count_params += up_params(256, 128)

    count_flops += conv_block_flops(256, 128, 64, 64)
    count_params += conv_block_params(256, 128)

    count_flops += up_flops(128, 64, 64, 128, 64, 128)
    count_params += up_params(128, 64)

    count_flops += conv_block_flops(64, 64, 128, 128)
    count_params += conv_block_params(64, 64)

    count_flops += 64 * 25 * 128 * 128 * 2 + 25 * 128 * 128
    count_params += 64 * 25

    return count_flops, count_params


def count_CFNet():
    count_flops, count_params = count_UNet()
    
    count_flops += HPFB_flops(25, 32, 128, 128)
    count_params += HPFB_params(25, 32)

    count_flops += down_flops(32, 32, 64, 64)
    count_params += down_params(32, 32)

    count_flops += HPFB_flops(32, 32, 64, 64)
    count_params += HPFB_params(32, 32)

    count_flops += down_flops(32, 32, 32, 32)
    count_params += down_params(32, 32)

    count_flops += HPFB_flops(32, 32, 32, 32)
    count_params += HPFB_params(32, 32)

    count_flops += up_flops(32, 32, 32, 64, 32, 64)
    count_params += up_params(32, 32)

    count_flops += HPFB_flops(64, 32, 64, 64)
    count_params += HPFB_params(64, 32)

    count_flops += up_flops(32, 32, 64, 128, 64, 128)
    count_params += up_params(32, 32)

    count_flops += HPFB_flops(64, 32, 128, 128)
    count_params += HPFB_params(64, 32)

    count_flops += 32 * 25 * 1 * 128 * 128 * 2 + 25 * 128 * 128
    count_params += 32 * 25 * 1

    return count_flops, count_params


def count_UNet_HP():
    n1 = 64
    filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
    count_flops = 0
    count_params = 0

    #conv1
    count_flops += HPFB_flops(25, 64, 128, 128)
    count_params += HPFB_params(25, 64)

    count_flops += down_flops(64, 64, 64, 64)
    count_params += down_params(64, 64)

    #conv2
    count_flops += HPFB_flops(64, 128, 64, 64)
    count_params += HPFB_params(64, 128)

    count_flops += down_flops(128, 128, 32, 32)
    count_params += down_params(128, 128)

    #conv3
    count_flops += HPFB_flops(128, 256, 32, 32)
    count_params += HPFB_params(128, 256)

    count_flops += down_flops(256, 256, 16, 16)
    count_params += down_params(256,256)

    #conv4
    count_flops += HPFB_flops(256, 512, 16, 16)
    count_params += HPFB_params(256, 512)

    count_flops += down_flops(512, 512, 8, 8)
    count_params += down_params(512, 512)

    #conv5
    count_flops += HPFB_flops(512, 1024, 8, 8)
    count_params += HPFB_params(512, 1024)

    #Up5
    count_flops += up_flops(1024, 512, 8, 16, 8, 16)
    count_params += up_params(1024, 512)

    #up conv5
    count_flops += HPFB_flops(1024, 512, 16, 16)
    count_params += HPFB_params(1024, 512)

    #Up4
    count_flops += up_flops(512, 256, 16, 32, 16, 32)
    count_params += up_params(512, 256)

    #up conv4
    count_flops += HPFB_flops(512, 256, 32, 32)
    count_params += HPFB_params(512, 256)

    count_flops += up_flops(256, 128, 32, 64, 32, 64)
    count_params += up_params(256, 128)

    count_flops += HPFB_flops(256, 128, 64, 64)
    count_params += HPFB_params(256, 128)

    count_flops += up_flops(128, 64, 64, 128, 64, 128)
    count_params += up_params(128, 64)

    count_flops += HPFB_flops(64, 64, 128, 128)
    count_params += HPFB_params(64, 64)

    count_flops += 64 * 25 * 128 * 128 * 2 + 25 * 128 * 128
    count_params += 64 * 25

    return count_flops, count_params


if __name__ == '__main__':
    # model = U_Net(25, 25)
    model = CF_Net(25, 25, 32, 8)
    # model = U_Net_HP_Final(25, 25)
    input = torch.zeros((1, 25, 128, 128))

    # flops = FlopCountAnalysis(model, input)
    # flops.total()
    # print(flops.total())
    # print(flop_count_table(flops))

    # print(count_UNet())
    flops, params = count_CFNet()
    # flops, params = count_UNet_HP()
    print(flops/1e9, params/1e6)