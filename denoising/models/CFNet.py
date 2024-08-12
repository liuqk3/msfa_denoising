from numpy import identity
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from PIL import ImageFilter
import cv2


def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


class HIN(nn.Module):
    def __init__(self, channel):
        super(HIN, self).__init__()
        self.IN = nn.InstanceNorm2d(channel//2)

    def forward(self, x):
        in_result = self.IN(x[:,:x.shape[1]//2,:,:])

        return torch.cat((in_result, x[:,x.shape[1]//2:,:,:]), dim=1)


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


class block_seq(nn.Module):
    def __init__(self, in_ch, out_ch, channel_per_group=8, draw=False):
        super(block_seq, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            FreqFilter(out_ch, 3, group=out_ch//channel_per_group, draw=draw),
            HIN(out_ch),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        self.conv_residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        return self.conv_residual(x) + self.conv(x)




class HPFB(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch, channel_per_group=8, draw=False):
        super(HPFB, self).__init__()
        
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.hpfilter = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            FreqFilter(out_ch, 3, group=out_ch//channel_per_group, draw=draw)
            # nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.HIN_relu = nn.Sequential(
            HIN(out_ch),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
     
        
        self.conv_residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True)
            

    def forward(self, x):
        out = self.conv_residual(x)+self.conv1(self.HIN_relu(self.conv0(x)+self.hpfilter(x)))
        return out



class FreqFilter(nn.Module):

    def __init__(self, in_channels, kernel_size, stride=1, pad_type='reflect', group=2, draw=False):
        super(FreqFilter, self).__init__()
        self.pad = get_pad_layer(pad_type)(kernel_size//2)
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(in_channels, group*kernel_size*kernel_size, kernel_size=kernel_size, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size*kernel_size)
        self.softmax = nn.Softmax(dim=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        self.draw = draw


    def forward(self, input):
        # x = input.clone()

        sigma = self.conv(self.pad(input))
        sigma = self.bn(sigma)
        sigma = self.softmax(sigma)


        # sigma = 1-sigma



        n,c,h,w = sigma.shape # kernel, shape = (N, k*k*g, H, W)

        sigma = sigma.reshape(n,1,c,h*w)# shape = (N, 1, k*k*g, H*W)

        n,c,h,w = input.shape
        x = F.unfold(self.pad(input), kernel_size=self.kernel_size).reshape((n,c,self.kernel_size*self.kernel_size,h*w))# shape = (N, C, k*k, H*W)

        n,c1,p,q = x.shape
        x = x.permute(1,0,2,3).reshape(self.group, c1//self.group, n, p, q).permute(2,0,1,3,4) # shape = (N, g, c/g, k*k, H*W)

        

        n,c2,p,q = sigma.shape
        sigma = sigma.permute(2,0,1,3).reshape((p//(self.kernel_size*self.kernel_size), self.kernel_size*self.kernel_size,n,c2,q)).permute(2,0,3,1,4)# shape = (N, g, 1, k*k, H*W)

        identity_kernel = torch.zeros_like(sigma).to(sigma.device)
        identity_kernel[:, :, :, self.kernel_size * self.kernel_size//2, :] = 1
        sigma = identity_kernel - sigma

        if self.draw:
            for i in range(4):
                draw_kernel(sigma[0, i, 0, :, :].reshape(-1, h, w))

        x = torch.sum(x*sigma, dim=3).reshape(n,c1,h,w)
        return x[:,:,torch.arange(h)%self.stride==0,:][:,:,:,torch.arange(w)%self.stride==0]

def draw_kernel(kernel):
    kernel = kernel.detach().cpu().numpy()
    import numpy as np
    kernel = np.transpose(kernel, (1,2,0))
    # kernel = np.array([0,0,0,0,1,0,0,0,0]) - kernel
    kernel = np.transpose(kernel, (2, 0, 1))
    var = np.var(kernel, axis=0)
    var = (var-np.min(var)) / (np.max(var)-np.min(var))
    # var = 1 - var
    from PIL import Image
    import os
    path = 'results_img_kernel_cfnet'
    if not os.path.exists(path):
        os.mkdir(path)
    num = len(os.listdir(path))

    Image.fromarray((var*255).astype(np.uint8)).save(os.path.join(path,str(num)+'.png'))

# def draw_kernel(kernel):
#     kernel = kernel.detach().cpu().numpy()
#     import numpy as np
#     var = np.var(kernel, axis=0)
#     var = (var-np.min(var)) / (np.max(var)-np.min(var))
#     # var = 1 - var
#     from PIL import Image
#     import os
#     path = 'results_img_kernel'
#     if not os.path.exists(path):
#         os.mkdir(path)
#     num = len(os.listdir(path))

#     Image.fromarray((var*255).astype(np.uint8)).save(os.path.join(path,str(num)+'.png'))


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


class CF_Net(nn.Module):
    def __init__(self, in_ch=25, out_ch=25, channels=64):
        super(CF_Net, self).__init__()

        self.coarse_stage = U_Net(in_ch, out_ch)
        self.FineBlock1 = conv_block(in_ch, channels)
        # self.FineBlock1 = block_seq(in_ch, channels, channel_per_groups, draw=True)
        # self.FineBlock1 = HPFB(in_ch, channels, channel_per_groups, draw=True)
        self.Down1 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=True)
        self.FineBlock2 = conv_block(channels, channels*2)
        # self.FineBlock2 = block_seq(channels, channels, channel_per_groups)
        # self.FineBlock2 = HPFB(channels, channels, channel_per_groups)
        self.Down2 = nn.Conv2d(channels*2, channels*2, kernel_size=4, stride=2, padding=1, bias=True)
        self.FineBlock3 = conv_block(channels*2, channels*4)
        # self.FineBlock3 = block_seq(channels, channels, channel_per_groups)
        # self.FineBlock3 = HPFB(channels, channels, channel_per_groups)
        self.Up2 = nn.ConvTranspose2d(channels*4, channels*2, kernel_size=2, stride=2, padding=0, bias=True)
        self.FineBlock4 = conv_block(channels*4, channels*2)
        # self.FineBlock4 = block_seq(channels*2, channels, channel_per_groups)
        # self.FineBlock4 = HPFB(channels*2, channels, channel_per_groups)
        self.Up1 = nn.ConvTranspose2d(channels*2, channels, kernel_size=2, stride=2, padding=0, bias=True)
        self.FineBlock5 = conv_block(channels*2, channels)
        # self.FineBlock5 = block_seq(channels*2, channels, channel_per_groups)
        # self.FineBlock5 = HPFB(channels*2, channels, channel_per_groups)

        self.conv = nn.Conv2d(channels, out_ch, kernel_size=1, padding=0, bias=True)

        # self.norm = nn.InstanceNorm2d(in_ch)

        # self.dmap_down1 = nn.Conv2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1, bias=True)
        # self.dmap_down2 = nn.Conv2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1, bias=True)


    def forward(self, x):
        coarse_x = self.coarse_stage(x)


        # dmap = coarse_x.clone()
        # dmap_list = []
        # for b in range(dmap.shape[0]):
        #     dmap_list.append(density(dmap[b,:,:,:]))
        # dmap = torch.cat(dmap_list, dim=0)
        # dmap = self.norm(dmap).detach()
        # dmap_d1 = self.dmap_down1(dmap)
        # dmap_d2 = self.dmap_down2(dmap_d1)
        dmap = density(x).detach()

        # input = torch.cat([x, dmap], dim=1)
        input = x

        e1 = self.FineBlock1(input)
        e2 = self.Down1(e1)

        e2 = self.FineBlock2(e2)
        # e2 = self.FineBlock2(torch.cat([e2, dmap_d1], dim=1))
        e3 = self.Down2(e2)

        e3 = self.FineBlock3(e3)
        # e3 = self.FineBlock3(torch.cat([e3, dmap_d2], dim=1))
        d2 = self.Up2(e3)
        d2 = torch.cat((d2, e2), dim=1)

        d2 = self.FineBlock4(d2)
        # d2 = self.FineBlock4(torch.cat([d2, dmap_d1], dim=1))
        d1 = self.Up1(d2)

        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.FineBlock5(d1)
        # d1 = self.FineBlock5(torch.cat([d1, dmap], dim=1))

        out = self.conv(d1) + coarse_x

        return out, coarse_x


class CF_Net_edge(nn.Module):
    def __init__(self, in_ch=25, out_ch=25, channels=64):
        super(CF_Net_edge, self).__init__()

        self.coarse_stage = U_Net(in_ch, out_ch)
        self.FineBlock1 = conv_block(in_ch*2, channels)
        # self.FineBlock1 = block_seq(in_ch, channels, channel_per_groups, draw=True)
        # self.FineBlock1 = HPFB(in_ch, channels, channel_per_groups, draw=True)
        self.Down1 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=True)
        self.FineBlock2 = conv_block(channels, channels*2)
        # self.FineBlock2 = block_seq(channels, channels, channel_per_groups)
        # self.FineBlock2 = HPFB(channels, channels, channel_per_groups)
        self.Down2 = nn.Conv2d(channels*2, channels*2, kernel_size=4, stride=2, padding=1, bias=True)
        self.FineBlock3 = conv_block(channels*2, channels*4)
        # self.FineBlock3 = block_seq(channels, channels, channel_per_groups)
        # self.FineBlock3 = HPFB(channels, channels, channel_per_groups)
        self.Up2 = nn.ConvTranspose2d(channels*4, channels*2, kernel_size=2, stride=2, padding=0, bias=True)
        self.FineBlock4 = conv_block(channels*4, channels*2)
        # self.FineBlock4 = block_seq(channels*2, channels, channel_per_groups)
        # self.FineBlock4 = HPFB(channels*2, channels, channel_per_groups)
        self.Up1 = nn.ConvTranspose2d(channels*2, channels, kernel_size=2, stride=2, padding=0, bias=True)
        self.FineBlock5 = conv_block(channels*2, channels)
        # self.FineBlock5 = block_seq(channels*2, channels, channel_per_groups)
        # self.FineBlock5 = HPFB(channels*2, channels, channel_per_groups)

        self.conv = nn.Conv2d(channels, out_ch, kernel_size=1, padding=0, bias=True)

        # self.norm = nn.InstanceNorm2d(in_ch)

        # self.dmap_down1 = nn.Conv2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1, bias=True)
        # self.dmap_down2 = nn.Conv2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1, bias=True)


    def forward(self, x):
        coarse_x = self.coarse_stage(x)


        dmap = coarse_x.clone()
        dmap_list = []
        for b in range(dmap.shape[0]):
            dmap_list.append(density(dmap[b,:,:,:]))
        dmap = torch.cat(dmap_list, dim=0)
        # dmap = self.norm(dmap).detach()
        # dmap_d1 = self.dmap_down1(dmap)
        # dmap_d2 = self.dmap_down2(dmap_d1)

        input = torch.cat([x, dmap], dim=1)

        e1 = self.FineBlock1(input)
        e2 = self.Down1(e1)

        # e2 = self.FineBlock2(e2)
        e2 = self.FineBlock2(torch.cat([e2, dmap_d1], dim=1))
        e3 = self.Down2(e2)

        # e3 = self.FineBlock3(e3)
        e3 = self.FineBlock3(torch.cat([e3, dmap_d2], dim=1))
        d2 = self.Up2(e3)
        d2 = torch.cat((d2, e2), dim=1)
        # d2 = self.FineBlock4(d2)
        d2 = self.FineBlock4(torch.cat([d2, dmap_d1], dim=1))
        d1 = self.Up1(d2)

        d1 = torch.cat((d1, e1), dim=1)
        # d1 = self.FineBlock5(d1)
        d1 = self.FineBlock5(torch.cat([d1, dmap], dim=1))

        out = self.conv(d1) + coarse_x

        return out, coarse_x


