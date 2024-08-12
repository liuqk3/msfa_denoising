from audioop import bias
from tkinter import W
from turtle import forward
from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F



class split_stride_conv(nn.Module):

    def __init__(self, channels=64, kernel_size=3):
        super(split_stride_conv, self).__init__()

        self.conv_list = nn.ModuleList([nn.Conv2d(1, channels, kernel_size=3, stride=3, padding=0, bias=True) for i in range(25)])

        self.kernel_size = kernel_size

        self.channels = channels

    def forward(self, x):
        
        results = torch.zeros((x.shape[0], self.channels, x.shape[2], x.shape[3])).to(x.device)

        for i in range(25):
            shape = x.shape
            h = i // 5
            w = i % 5
            shift_h = 1 - h
            shift_w = 1 - w
            band = torch.roll(x, (shift_h, shift_w), dims=(2, 3))
            band = F.unfold(band, kernel_size=3, padding=0, stride=5)

            band = F.fold(band, (shape[2]//5*3, shape[3]//5*3), kernel_size=3, padding=0, stride=3)
            band = self.conv_list[0](band)

            results[:, :, h::5, w::5] = band

        return results


# class split_stride_conv(nn.Module):
#     def __init__(self, channels=64):
#         super(split_stride_conv, self).__init__()

#         self.conv_list = nn.ModuleList([split_stride_conv_single(kernel_size=3) for i in range(channels)])

#     def forward(self, x):
#         results = []
#         for i in range(len(self.conv_list)):
#             results.append(self.conv_list[i](x))

#         out = torch.cat(results, dim=1)

#         return out


class stride_conv_relu(nn.Module):
    def __init__(self, channel=64):
        super(stride_conv_relu, self).__init__()

        self.prelu = nn.PReLU()
        self.conv = split_stride_conv(channel)
        
    def forward(self, x):
        return self.prelu(self.conv(x))



class residual_block(nn.Module):

    def __init__(self, channels=1):
        super(residual_block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU()
        )

    def forward(self, x):
        return x + self.conv2(self.conv1(x))



def pack(x):
    results = torch.zeros((x.shape[0], 1, x.shape[2]*5, x.shape[3]*5)).to(x.device)
    for i in range(25):
        pattern_r = i // 5
        pattern_c = i % 5

        results[:, :, pattern_r::5, pattern_c::5] = x[:, i, :, :]
    return results


def unpack(x):
    results = torch.zeros((x.shape[0], 25, x.shape[2]//5, x.shape[3]//5)).to(x.device)
    for i in range(25):
        pattern_r = i // 5
        pattern_c = i % 5
        results[:, i, :, :] = x[:, 0, pattern_r::5, pattern_c::5]

    return results




class Pan_MSFA(nn.Module):
    
    def __init__(self, channels=64):
        super(Pan_MSFA, self).__init__()

        self.stride_conv = stride_conv_relu()

        self.conv_list = nn.ModuleList([residual_block(channels) for i in range(9)])

        self.conv = nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)


    def forward(self, x):

        x = pack(x)

        e = self.stride_conv(x)

        for i in range(9):
            e = self.conv_list[i](e)

        e = self.conv(e)

        out = x - e
        out = unpack(out)

        return out