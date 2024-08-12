import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
from PIL import Image
from PIL import ImageFilter
import numpy as np
import cv2

if __name__ == "__main__":
    import common
else:
    from models import common


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


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class AttFFN(nn.Module):
    def __init__(self, out_ch, patchsize=8, dim=1024, mlp_dim=2048, dropout=0.1):
        super(AttFFN, self).__init__()

        patch_dim = out_ch * patchsize * patchsize
        self.attention = PreNorm(dim, Attention(dim, heads = 8, dim_head = 64, dropout = 0.))

        self.to_patch_embedding1 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patchsize, p2 = patchsize),
            nn.Linear(patch_dim, dim),
        )
        self.unembedding1 = nn.Linear(dim, patch_dim)

        self.patchsize = patchsize

    def forward(self, x):

        B, C, H, W = x.shape

        embedding1 = self.to_patch_embedding1(x)
        attn_out = self.attention(embedding1)
        attn_out = self.unembedding1(attn_out)
        attn_out = attn_out.view(x.shape)
        x = x + attn_out

        return x


class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)

        return x


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


class GlobalFilter(nn.Module):
    def __init__(self, dim, h=8, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)

        return x


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
            # nn.InstanceNorm2d(out_ch),
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



class HPFB_v2(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch, channel_per_group=8):
        super(HPFB_v2, self).__init__()
        
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.hpfilter = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            FreqFilter(out_ch, 3, group=out_ch//channel_per_group)
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
            

    def forward(self, x, dmap):

        x = torch.cat((x, dmap), dim=1)
        out = self.conv_residual(x)+self.conv1(self.HIN_relu(self.conv0(x)+self.hpfilter(x)))
        return out


class HPFB_v3(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch, channel_per_group=8):
        super(HPFB_v3, self).__init__()
        
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_ch+1, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.hpfilter = nn.Sequential(
            nn.Conv2d(in_ch+1, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            FreqFilter(out_ch, 3, group=out_ch//channel_per_group)
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
        dmap = torch.mean(x, dim=1)
        dmap = density(dmap)

        x = torch.cat((x, dmap), dim=1)
        out = self.conv_residual(x)+self.conv1(self.HIN_relu(self.conv0(x)+self.hpfilter(x)))
        return out


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
    path = 'results_img_kernel_unethp_HIN'
    if not os.path.exists(path):
        os.mkdir(path)
    num = len(os.listdir(path))

    Image.fromarray((var*255).astype(np.uint8)).save(os.path.join(path,str(num)+'.png'))


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

        if self.draw:
            for i in range(8):
                draw_kernel(sigma[0][i*self.kernel_size * self.kernel_size:(i+1)*self.kernel_size * self.kernel_size])

        n,c,h,w = sigma.shape # kernel, shape = (N, k*k*g, H, W)

        sigma = sigma.reshape(n,1,c,h*w)# shape = (N, 1, k*k*g, H*W)

        n,c,h,w = input.shape
        x = F.unfold(self.pad(input), kernel_size=self.kernel_size).reshape((n,c,self.kernel_size*self.kernel_size,h*w))# shape = (N, C, k*k, H*W)

        n,c1,p,q = x.shape
        x = x.permute(1,0,2,3).reshape(self.group, c1//self.group, n, p, q).permute(2,0,1,3,4) # shape = (N, g, c/g, k*k, H*W)

        n,c2,p,q = sigma.shape
        sigma = sigma.permute(2,0,1,3).reshape((p//(self.kernel_size*self.kernel_size), self.kernel_size*self.kernel_size,n,c2,q)).permute(2,0,3,1,4)# shape = (N, g, 1, k*k, H*W)



        x = torch.sum(x*sigma, dim=3).reshape(n,c1,h,w)
        return input - x[:,:,torch.arange(h)%self.stride==0,:][:,:,:,torch.arange(w)%self.stride==0]



def density( x):
    x = torch.clamp(x*255,0.,255.).detach()
    c,w,h = x.shape
    
    im= np.array(x[0].cpu()).astype(np.uint8)
    im = Image.fromarray(im)
    im_blur = im.filter(ImageFilter.GaussianBlur(radius=3))
    im_minus  = abs(np.array(im).astype(np.float)-np.array(im_blur).astype(np.float))
    im_minus = np.uint8(im_minus)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩形结构
    im_sum = torch.from_numpy(cv2.dilate(im_minus, kernel).astype(np.float))
    im_sum = im_sum.unsqueeze(0)
    #print(im_sum.shape)    
    for i in range(1,c):
        im= np.array(x[i].cpu()).astype(np.uint8)
        im = Image.fromarray(im)
        im_blur = im.filter(ImageFilter.GaussianBlur(radius=5))
        im_minus  = abs(np.array(im).astype(np.float)-np.array(im_blur).astype(np.float))
        im_minus = np.uint8(im_minus)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩形结构
        im_minus = cv2.dilate(im_minus, kernel).astype(np.float)
        im_minus = torch.from_numpy(im_minus).unsqueeze(0)
        #print(im_minus.shape)
        im_sum = torch.cat([im_sum,im_minus] , 0 )
    return (im_sum/255).unsqueeze(0).float().cuda()


class block_seq(nn.Module):
    def __init__(self, in_ch, out_ch, channel_per_group=8):
        super(block_seq, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            FreqFilter(out_ch, 3, group=out_ch//channel_per_group),
            HIN(out_ch),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        self.conv_residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        return self.conv_residual(x) + self.conv(x)


class U_Net_HP(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=34, out_ch=34, channel_per_group=8):
        super(U_Net_HP, self).__init__()

        channels = 64
        
        self.Down1 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=True)
        self.Down2 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=True)
        self.Down3 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=True)
        self.Down4 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=True)

        self.Conv1 = HPFB(in_ch, channels, channel_per_group, draw=True)
        self.Conv2 = HPFB(channels, channels, channel_per_group)
        self.Conv3 = HPFB(channels, channels, channel_per_group)
        self.Conv4 = HPFB(channels, channels, channel_per_group)
        self.Conv5 = HPFB(channels, channels, channel_per_group)
        # self.Conv1 = HPFB_v2(in_ch*2, channels, channel_per_group)
        # self.Conv2 = HPFB_v2(channels*2, channels, channel_per_group)
        # self.Conv3 = HPFB_v2(channels*2, channels, channel_per_group)
        # self.Conv4 = HPFB_v2(channels*2, channels, channel_per_group)
        # self.Conv5 = HPFB_v2(channels*2, channels, channel_per_group)
        # self.Conv1 = block_seq(in_ch, channels, channel_per_group)
        # self.Conv2 = block_seq(channels, channels, channel_per_group)
        # self.Conv3 = block_seq(channels, channels, channel_per_group)
        # self.Conv4 = block_seq(channels, channels, channel_per_group)
        # self.Conv5 = block_seq(channels, channels, channel_per_group)

        self.Up5 = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2, padding=0, bias=True)
        self.Up4 = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2, padding=0, bias=True)
        self.Up3 = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2, padding=0, bias=True)
        self.Up2 = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2, padding=0, bias=True)

        self.Up_conv5 = HPFB(channels*2, channels, channel_per_group)
        self.Up_conv4 = HPFB(channels*2, channels, channel_per_group)
        self.Up_conv3 = HPFB(channels*2, channels, channel_per_group)
        self.Up_conv2 = HPFB(channels*2, channels, channel_per_group)
        # self.Up_conv2 = conv_block(channels*2, channels)
        # self.Up_conv5 = block_seq(channels*2, channels, channel_per_group)
        # self.Up_conv4 = block_seq(channels*2, channels, channel_per_group)
        # self.Up_conv3 = block_seq(channels*2, channels, channel_per_group)
        # self.Up_conv2 = block_seq(channels*2, channels, channel_per_group)

        self.Conv = nn.Conv2d(channels, out_ch, kernel_size=1, stride=1, padding=0)

        # self.Down1_density = nn.Conv2d(in_ch, channels, kernel_size=4, stride=2, padding=1, bias=True)
        # self.Down2_density = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=True)
        # self.Down3_density = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=True)
        # self.Down4_density = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=True)

    def forward(self, x):

        dmap = x.clone()
        dmap_list = []
        for b in range(dmap.shape[0]):
            dmap_list.append(density(dmap[b,:,:,:]))
        # dmap = torch.cat(dmap_list, dim=0)
        # dmap_d1 = self.Down1_density(dmap)
        # dmap_d2 = self.Down2_density(dmap_d1)
        # dmap_d3 = self.Down3_density(dmap_d2)
        # dmap_d4 = self.Down4_density(dmap_d3)

        e1 = self.Conv1(x)
        # e1 = self.Conv1(x, dmap)

        e2 = self.Down1(e1)
        e2 = self.Conv2(e2)
        # e2 = self.Conv2(e2, dmap_d1)

        e3 = self.Down2(e2)
        e3 = self.Conv3(e3)
        # e3 = self.Conv3(e3, dmap_d2)

        e4 = self.Down3(e3)
        e4 = self.Conv4(e4)
        # e4 = self.Conv4(e4, dmap_d3)

        e5 = self.Down4(e4)
        e5 = self.Conv5(e5)
        # e5 = self.Conv5(e5, dmap_d4)

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


        return out+x


