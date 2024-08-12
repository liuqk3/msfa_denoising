from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import numpy as np
from PIL import Image
from PIL import ImageFilter
import cv2

if __name__ == "__main__":
    import common
else:
    from models import common


class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True))
        
        self.conv_residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True)

    def forward(self, x):

        x = self.conv(x) + self.conv_residual(x)
        return x

class down_conv(nn.Module):
    """
    Down Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(down_conv, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
            conv_block(out_ch, out_ch)
        )

    def forward(self, x):
        x = self.down(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x




# def density( x):
#     x = torch.clamp(x*255,0.,255.).detach()
#     c,w,h = x.shape
    
#     im= np.array(x[0].cpu()).astype(np.uint8)
#     im = Image.fromarray(im)
#     im_blur = im.filter(ImageFilter.GaussianBlur(radius=3))
#     im_minus  = abs(np.array(im).astype(np.float)-np.array(im_blur).astype(np.float))
#     im_minus = np.uint8(im_minus)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩形结构
#     im_sum = torch.from_numpy(cv2.dilate(im_minus, kernel).astype(np.float))
#     im_sum = im_sum.unsqueeze(0)
#     #print(im_sum.shape)    
#     for i in range(1,c):
#         im= np.array(x[i].cpu()).astype(np.uint8)
#         im = Image.fromarray(im)
#         im_blur = im.filter(ImageFilter.GaussianBlur(radius=5))
#         im_minus  = abs(np.array(im).astype(np.float)-np.array(im_blur).astype(np.float))
#         im_minus = np.uint8(im_minus)
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩形结构
#         im_minus = cv2.dilate(im_minus, kernel).astype(np.float)
#         im_minus = torch.from_numpy(im_minus).unsqueeze(0)
#         #print(im_minus.shape)
#         im_sum = torch.cat([im_sum,im_minus] , 0 )
#     return (im_sum/255).unsqueeze(0).float().cuda()


def density(x):
    x = torch.clamp(x*255,0.,255.).detach()
    x = torch.mean(x, dim=1)
    b,w,h = x.shape
    
    im_sum = []
    for i in range(b):
        im = np.array(x[i].cpu()).astype(np.uint8)
        im_blur = cv2.GaussianBlur(im, (5,5), 0)
        im_minus = abs(im.astype(np.float) - im_blur.astype(np.float)).astype(np.uint8)
        im_edge = cv2.GaussianBlur(im_minus, (5,5), 0).astype(np.float)
        im_edge = (im_edge - np.min(im_edge)) / (np.max(im_edge) - np.min(im_edge))
        im_edge = torch.from_numpy(im_edge)
        im_sum.append(im_edge.unsqueeze(0).unsqueeze(0))

    im_sum = torch.cat(im_sum, dim=0).float().cuda()
    
    return im_sum


# class DAM(nn.Module):
#     def __init__(self, channel):
#         super(DAM, self).__init__()
#         self.conv = nn.Conv2d(channel*2, channel, kernel_size=1, stride=1, padding=0, bias=True)

#     def forward(self, x, dmap):
#         attn = x * dmap
#         attn = torch.cat([x, attn], dim = 1)
#         return self.conv(attn)


def DAM(x, dmap):
    attn = x * dmap
    attn = x + attn

    return attn



class U_Net_edge(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=34, out_ch=34):
        super(U_Net_edge, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Down1 = nn.Conv2d(filters[0], filters[0], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down2 = nn.Conv2d(filters[1], filters[1], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down3 = nn.Conv2d(filters[2], filters[2], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down4 = nn.Conv2d(filters[3], filters[3], kernel_size=4, stride=2, padding=1, bias=True)

        self.Conv1 = conv_block(in_ch+1, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        # self.norm = nn.InstanceNorm2d(in_ch)

    def forward(self, x):

        dmap = x.clone()
        dmap = density(dmap).detach()

        e1 = self.Conv1(torch.cat([x, dmap], dim=1))

        e2 = self.Down1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Down2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Down3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Down4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #d1 = self.active(out)

        return out+x



class U_Net_edge_v2(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=34, out_ch=34):
        super(U_Net_edge_v2, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Down1 = nn.Conv2d(filters[0], filters[0], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down2 = nn.Conv2d(filters[1], filters[1], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down3 = nn.Conv2d(filters[2], filters[2], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down4 = nn.Conv2d(filters[3], filters[3], kernel_size=4, stride=2, padding=1, bias=True)

        self.Conv1 = conv_block(in_ch+1, filters[0])
        self.Conv2 = conv_block(filters[0]+1, filters[1])
        self.Conv3 = conv_block(filters[1]+1, filters[2])
        self.Conv4 = conv_block(filters[2]+1, filters[3])
        self.Conv5 = conv_block(filters[3]+1, filters[4])

        self.Up5 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv5 = conv_block(filters[4]+1, filters[3])

        self.Up4 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv4 = conv_block(filters[3]+1, filters[2])

        self.Up3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv3 = conv_block(filters[2]+1, filters[1])

        self.Up2 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv2 = conv_block(filters[1]+1, filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        # self.norm = nn.InstanceNorm2d(in_ch)

        self.dmap_down1 = nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1, bias=True)
        self.dmap_down2 = nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1, bias=True)
        self.dmap_down3 = nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1, bias=True)
        self.dmap_down4 = nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1, bias=True)

    def forward(self, x):

        dmap = x.clone()
        dmap = density(dmap).detach()
        dmap_d1 = self.dmap_down1(dmap)
        dmap_d2 = self.dmap_down2(dmap_d1)
        dmap_d3 = self.dmap_down3(dmap_d2)
        dmap_d4 = self.dmap_down4(dmap_d3)

        e1 = self.Conv1(torch.cat([x, dmap], dim=1))

        e2 = self.Down1(e1)
        e2 = self.Conv2(torch.cat([e2, dmap_d1], dim=1))

        e3 = self.Down2(e2)
        e3 = self.Conv3(torch.cat([e3, dmap_d2], dim=1))

        e4 = self.Down3(e3)
        e4 = self.Conv4(torch.cat([e4, dmap_d3], dim=1))

        e5 = self.Down4(e4)
        e5 = self.Conv5(torch.cat([e5, dmap_d4], dim=1))

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(torch.cat([d5, dmap_d3], dim=1))

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(torch.cat([d4, dmap_d2], dim=1))

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(torch.cat([d3, dmap_d1], dim=1))

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(torch.cat([d2, dmap], dim=1))

        out = self.Conv(d2)

        #d1 = self.active(out)

        return out+x


class U_Net_edge_v3(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=34, out_ch=34):
        super(U_Net_edge_v3, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Down1 = nn.Conv2d(filters[0], filters[0], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down2 = nn.Conv2d(filters[1], filters[1], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down3 = nn.Conv2d(filters[2], filters[2], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down4 = nn.Conv2d(filters[3], filters[3], kernel_size=4, stride=2, padding=1, bias=True)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        # self.norm = nn.InstanceNorm2d(in_ch)

        self.dmap_down1 = nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1, bias=True)
        self.dmap_down2 = nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1, bias=True)
        self.dmap_down3 = nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1, bias=True)
        self.dmap_down4 = nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1, bias=True)


    def forward(self, x):

        dmap = x.clone()
        dmap = density(dmap).detach()
        dmap_d1 = self.dmap_down1(dmap)
        dmap_d2 = self.dmap_down2(dmap_d1)
        dmap_d3 = self.dmap_down3(dmap_d2)
        dmap_d4 = self.dmap_down4(dmap_d3)

        
        # e1 = self.Conv1(e1)
        e1 = DAM(x, dmap)
        e1 = self.Conv1(e1)

        e2 = self.Down1(e1)
        e2 = DAM(e2, dmap_d1)
        e2 = self.Conv2(e2)

        e3 = self.Down2(e2)
        e3 = DAM(e3, dmap_d2)
        e3 = self.Conv3(e3)


        e4 = self.Down3(e3)
        e4 = DAM(e4, dmap_d3)
        e4 = self.Conv4(e4)

        e5 = self.Down4(e4)
        e5 = DAM(e5, dmap_d4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5) # 512
        d5 = DAM(d5, dmap_d3)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)# 256
        d4 = DAM(d4, dmap_d2)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)# 128
        d3 = DAM(d3, dmap_d1)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3) # 64
        d2 = DAM(d2, dmap)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #d1 = self.active(out)

        return out+x