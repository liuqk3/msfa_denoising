from numpy import identity
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from PIL import ImageFilter
import cv2


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
        out = self.conv(x) + self.conv_residual(x)
        return out



class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=34, out_ch=34):
        super(U_Net, self).__init__()

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

    def forward(self, x):

        e1 = self.Conv1(x)

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


class CF_Net(nn.Module):
    def __init__(self, in_ch=25, out_ch=25, channels=64):
        super(CF_Net, self).__init__()

        self.coarse_stage = U_Net(in_ch, out_ch)
        self.FineBlock1 = conv_block(in_ch, channels)
        self.Down1 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=True)
        self.FineBlock2 = conv_block(channels, channels*2)
        self.Down2 = nn.Conv2d(channels*2, channels*2, kernel_size=4, stride=2, padding=1, bias=True)
        self.FineBlock3 = conv_block(channels*2, channels*4)
        self.Up2 = nn.ConvTranspose2d(channels*4, channels*2, kernel_size=2, stride=2, padding=0, bias=True)
        self.FineBlock4 = conv_block(channels*4, channels*2)
        self.Up1 = nn.ConvTranspose2d(channels*2, channels, kernel_size=2, stride=2, padding=0, bias=True)
        self.FineBlock5 = conv_block(channels*2, channels)

        self.conv = nn.Conv2d(channels, out_ch, kernel_size=1, padding=0, bias=True)


    def forward(self, x):
        dmap = density(x).detach()

        coarse_x = self.coarse_stage(x)

        dmap = density(coarse_x).detach()

        e1 = self.FineBlock1(coarse_x)

        e2 = self.Down1(e1)
        e2 = self.FineBlock2(e2)

        e3 = self.Down2(e2)
        e3 = self.FineBlock3(e3)

        d2 = self.Up2(e3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.FineBlock4(d2)

        d1 = self.Up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.FineBlock5(d1)

        out = self.conv(d1) + coarse_x

        return out, coarse_x


class CF_Net_edge_v1(nn.Module):
    def __init__(self, in_ch=25, out_ch=25, channels=64):
        super(CF_Net_edge_v1, self).__init__()

        self.coarse_stage = U_Net(in_ch, out_ch)
        self.FineBlock1 = conv_block(in_ch+1, channels)
        self.Down1 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=True)
        self.FineBlock2 = conv_block(channels, channels*2)
        self.Down2 = nn.Conv2d(channels*2, channels*2, kernel_size=4, stride=2, padding=1, bias=True)
        self.FineBlock3 = conv_block(channels*2, channels*4)
        self.Up2 = nn.ConvTranspose2d(channels*4, channels*2, kernel_size=2, stride=2, padding=0, bias=True)
        self.FineBlock4 = conv_block(channels*4, channels*2)
        self.Up1 = nn.ConvTranspose2d(channels*2, channels, kernel_size=2, stride=2, padding=0, bias=True)
        self.FineBlock5 = conv_block(channels*2, channels)

        self.conv = nn.Conv2d(channels, out_ch, kernel_size=1, padding=0, bias=True)



    def forward(self, x):
        coarse_x = self.coarse_stage(x)

        dmap = coarse_x.clone()
        dmap = density(dmap).detach()

        input = torch.cat([coarse_x, dmap], dim=1)

        e1 = self.FineBlock1(input)

        e2 = self.Down1(e1)
        e2 = self.FineBlock2(e2)

        e3 = self.Down2(e2)
        e3 = self.FineBlock3(e3)

        d2 = self.Up2(e3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.FineBlock4(d2)
        d1 = self.Up1(d2)

        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.FineBlock5(d1)

        out = self.conv(d1) + coarse_x

        return out, coarse_x


class CF_Net_edge_v2(nn.Module):
    def __init__(self, in_ch=25, out_ch=25, channels=64):
        super(CF_Net_edge_v2, self).__init__()

        self.coarse_stage = U_Net(in_ch, out_ch)
        self.FineBlock1 = conv_block(in_ch+1, channels)
        self.Down1 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=True)
        self.FineBlock2 = conv_block(channels+1, channels*2)
        self.Down2 = nn.Conv2d(channels*2, channels*2, kernel_size=4, stride=2, padding=1, bias=True)
        self.FineBlock3 = conv_block(channels*2+1, channels*4)
        self.Up2 = nn.ConvTranspose2d(channels*4, channels*2, kernel_size=2, stride=2, padding=0, bias=True)
        self.FineBlock4 = conv_block(channels*4+1, channels*2)
        self.Up1 = nn.ConvTranspose2d(channels*2, channels, kernel_size=2, stride=2, padding=0, bias=True)
        self.FineBlock5 = conv_block(channels*2+1, channels)

        self.conv = nn.Conv2d(channels, out_ch, kernel_size=1, padding=0, bias=True)

        self.dmap_down1 = nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1, bias=True)
        self.dmap_down2 = nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1, bias=True)



    def forward(self, x):
        coarse_x = self.coarse_stage(x)

        dmap = coarse_x.clone()
        dmap = density(dmap).detach()
        dmap_down1 = self.dmap_down1(dmap)
        dmap_down2 = self.dmap_down2(dmap_down1)

        input = torch.cat([coarse_x, dmap], dim=1)

        e1 = self.FineBlock1(input)

        e2 = self.Down1(e1)
        e2 = self.FineBlock2(torch.cat([e2, dmap_down1], dim=1))

        e3 = self.Down2(e2)
        e3 = self.FineBlock3(torch.cat([e3, dmap_down2], dim=1))

        d2 = self.Up2(e3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.FineBlock4(torch.cat([d2, dmap_down1], dim=1))
        d1 = self.Up1(d2)

        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.FineBlock5(torch.cat([d1, dmap], dim=1))

        out = self.conv(d1) + coarse_x

        return out, coarse_x


def DAM(x, dmap):
    attn = x * dmap
    attn = x + attn

    return attn


class CF_Net_edge_v3(nn.Module):
    def __init__(self, in_ch=25, out_ch=25, channels=64):
        super(CF_Net_edge_v3, self).__init__()

        self.coarse_stage = U_Net(in_ch, out_ch)
        self.FineBlock1 = conv_block(in_ch, channels)
        self.Down1 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=True)
        self.FineBlock2 = conv_block(channels, channels*2)
        self.Down2 = nn.Conv2d(channels*2, channels*2, kernel_size=4, stride=2, padding=1, bias=True)
        self.FineBlock3 = conv_block(channels*2, channels*4)
        self.Up2 = nn.ConvTranspose2d(channels*4, channels*2, kernel_size=2, stride=2, padding=0, bias=True)
        self.FineBlock4 = conv_block(channels*4, channels*2)
        self.Up1 = nn.ConvTranspose2d(channels*2, channels, kernel_size=2, stride=2, padding=0, bias=True)
        self.FineBlock5 = conv_block(channels*2, channels)

        self.conv = nn.Conv2d(channels, out_ch, kernel_size=1, padding=0, bias=True)

        self.dmap_down1 = nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1, bias=True)
        self.dmap_down2 = nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1, bias=True)



    def forward(self, x):
        coarse_x = self.coarse_stage(x)

        dmap = coarse_x.clone()
        dmap = density(dmap).detach()
        dmap_down1 = self.dmap_down1(dmap)
        dmap_down2 = self.dmap_down2(dmap_down1)

        
        input = DAM(coarse_x, dmap)
        e1 = self.FineBlock1(input)

        e2 = self.Down1(e1)
        e2 = self.FineBlock2(DAM(e2, dmap_down1))

        e3 = self.Down2(e2)
        e3 = self.FineBlock3(DAM(e3, dmap_down2))

        d2 = self.Up2(e3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.FineBlock4(DAM(d2, dmap_down1))
        d1 = self.Up1(d2)

        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.FineBlock5(DAM(d1, dmap))

        out = self.conv(d1) + coarse_x

        return out, coarse_x