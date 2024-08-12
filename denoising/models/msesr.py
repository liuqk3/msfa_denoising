if __name__ == '__main__':
    import common
else:
    from models import common

import torch
import torch.nn as nn
from torch.autograd import Variable
# from torch.utils.checkpoint import checkpoint_sequential
import numpy as np
import os

def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        # nn.init.kaiming_uniform_(m.weight.data)
        m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        # nn.init.xavier_uniform_(m.weight.data)
        # nn.init.kaiming_uniform_(m.weight.data)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        # nn.init.kaiming_uniform_(m.weight.data)
        m.bias.data.zero_()

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight
 
    def forward(self, x, mode=[2,3]):
        batch_size = x.size()[0]
        c_x = x.size()[1]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_c = self._tensor_size(x[:,1:,:,:])
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        # h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        # w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        # return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

        c_tv, h_tv, w_tv = 0, 0, 0
        if 1 in mode:
            c_tv = torch.abs((x[:,1:,:,:] - x[:,:c_x-1,:,:])).sum() / count_c
        if 2 in mode:
            h_tv = torch.abs((x[:,:,1:,:] - x[:,:,:h_x-1,:])).sum() / count_h
        if 3 in mode:
            w_tv = torch.abs((x[:,:,:,1:] - x[:,:,:,:w_x-1])).sum() / count_w
        return self.TVLoss_weight * (c_tv+h_tv+w_tv) / batch_size
 
    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class LowrankLoss(nn.Module):
    def __init__(self, LowrankLoss_weight=1, sigma=1e-3, eps=1e-6):
        super(LowrankLoss, self).__init__()
        self.LowrankLoss_weight = LowrankLoss_weight
        self.sigma = sigma
        self.eps = eps

    def forward(self, x):
        x = x.view(x.size()[1], x.size()[2]*x.size()[3]).permute(1, 0)
        u,s,v = torch.svd(x)
        return s.sum()

        # s_max = torch.clamp(s.pow(2).div(torch.sum(x, dim=0)), min=0)
        # s_max = s.pow(2).div(torch.sum(x, dim=0))
        # weight = self.sigma * torch.pow(s_max.sqrt()+self.eps, -1)
        return s.mul(weight).sum()

class MemoryBlock(nn.Module):
    def __init__(self, n_resblocks, n_memblocks, n_feats, kernel_size, conv=common.default_conv, se_reduction=None, if_last=False, out_channels=0):
        super(MemoryBlock, self).__init__()
        self.memory_unit = nn.ModuleList(
            [common.SEResBlock(conv, n_feats, kernel_size, se_reduction=se_reduction) for _ in range(n_resblocks)]
        )
        if out_channels and if_last:
            self.gate_unit = conv((n_resblocks+n_memblocks) * n_feats, out_channels, 1)
        else:
            self.gate_unit = conv((n_resblocks+n_memblocks) * n_feats, n_feats, 1)
        self.if_last = if_last

        self.act = nn.ReLU(True)
        self.out_channels = out_channels

    def forward(self, x, ys):
        xs = []
        for layer in self.memory_unit:
            x = layer(x)
            xs.append(x)

        if self.out_channels and self.if_last:
            gate_out = self.gate_unit(torch.cat(xs+ys, 1))
        else:
            gate_out = self.act(self.gate_unit(torch.cat(xs+ys, 1)))
        ys.append(gate_out)
        return gate_out

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()
        self.dconv_down1 = common.double_conv(in_channels, 32)
        self.dconv_down2 = common.double_conv(32, 64)
        self.dconv_down3 = common.double_conv(64, 128)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.dconv_up2 = common.double_conv(64+64, 64)
        self.dconv_up1 = common.double_conv(32+32, 32)

        self.conv_last = nn.Conv2d(32, out_channels, 1)
        self.act_last = nn.Tanh()
    
    def forward(self, x):
        inputs = x
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)

        x = self.upsample2(conv3)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample1(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)
        x = self.act_last(x)
        
        return x + inputs

class Denoiser_v1(nn.Module):
    def __init__(self, n_resblocks, in_channels, n_feats, kernel_size, conv=common.default_conv, se_reduction=None):
        super(Denoiser_v1, self).__init__()
        self.up_features = conv(in_channels, n_feats, kernel_size)
        self.down_features = conv(n_feats, in_channels, kernel_size)
        self.res_blocks = nn.ModuleList(
            [common.SEResBlock(conv, n_feats, kernel_size, se_reduction=se_reduction) for _ in range(n_resblocks)]
        )
        self.act = nn.ReLU(True)
    
    def forward(self, x):
        y = self.act(self.up_features(x))
        for block in self.res_blocks:
            y = block(y)
        y = self.down_features(y)

        return y

# class Denoiser_EMA(nn.Module):
#     def __init__(self, in_channels, n_feats, kernel_size, conv=common.default_conv, se_reduction=None):
#         super(Denoiser_EMA, self).__init__()
#         self.ema = common.EMAU(in_channels, n_feats)
#         self.up_features = conv(in_channels, n_feats, kernel_size, groups=in_channels)
#         self.down_features = conv(n_feats, in_channels, kernel_size, groups=in_channels)
#         self.res_block1 = common.SEResBlock(conv, n_feats, kernel_size, groups=in_channels, se_reduction=se_reduction)
#         self.res_block2 = common.SEResBlock(conv, n_feats, kernel_size, groups=in_channels, se_reduction=se_reduction)
    
#     def forward(self, x):
#         res = x
#         x = self.up_features(x)
#         x = self.res_block1(x)
#         x = self.ema(x)
#         x = self.res_block2(x)
#         x = self.down_features(x)
#         return x+res

class Denoiser_EMA(nn.Module):
    def __init__(self, in_channels, n_feats, kernel_size, cube_size=[3,3], conv=common.default_conv, se_reduction=None):
        super(Denoiser_EMA, self).__init__()
        groups = in_channels
        dilation = 3
        padding = (dilation*cube_size[0]-dilation)//2
        self.sf = 4
        self.channels = n_feats//in_channels

        self.lr = common.LowRankSTLayer_dilation(n_feats//in_channels, n_feats//in_channels//2, cube_size[0], cube_size[1], dilation=dilation, padding=padding, duration_padding=cube_size[1]//2)

        self.down = common.PixelUnShuffle(upscale_factor=self.sf)
        self.up = nn.PixelShuffle(upscale_factor=self.sf)
        self.head = nn.Conv2d(in_channels*self.sf*self.sf, n_feats, kernel_size, groups=groups, padding=kernel_size//2)
        self.tail = nn.Conv2d(n_feats, in_channels*self.sf*self.sf, kernel_size, groups=groups, padding=kernel_size//2)
        self.res_block1 = common.SEResBlock(conv, n_feats, kernel_size, groups=groups, se_reduction=se_reduction)
        self.res_block2 = common.SEResBlock(conv, n_feats, kernel_size, groups=groups, se_reduction=se_reduction)

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        batch, duration, height, width = x.size()

        res = x
        x = self.down(x)
        x = self.relu(self.head(x))
        x = self.res_block1(x)
        x = x.view(batch, duration, self.channels, height//self.sf, width//self.sf).permute(0,2,1,3,4).contiguous()
        x = self.lr(x)
        x = x.permute(0,2,1,3,4).contiguous().view(batch, duration*self.channels, height//self.sf, width//self.sf)
        x = self.res_block2(x)
        x = self.tail(x)
        x = self.up(x)

        return x+res

class Denoiser_LR(nn.Module):
    def __init__(self, in_channels, n_feats, n_resblocks, kernel_size, cube_size=[5,3], conv=common.default_conv, se_reduction=None):
        super(Denoiser_LR, self).__init__()
        groups = 1
        dilation = 3
        padding = (dilation*cube_size[0]-dilation)//2
        self.sf = 2
        # self.channels = 8*n_feats//in_channels
        self.channels = 8*n_feats

        self.head = nn.Conv2d(in_channels, n_feats, kernel_size, groups=groups, padding=kernel_size//2)

        self.res_block1_1 = nn.Sequential(*[common.SEResBlock(conv, n_feats, kernel_size, groups=groups, se_reduction=se_reduction) for _ in range(n_resblocks)])
        self.down1 = nn.Conv2d(n_feats, n_feats*2, 4, stride=2, groups=groups, padding=1)
        self.res_block2_1 = nn.Sequential(*[common.SEResBlock(conv, n_feats*2, kernel_size, groups=groups, se_reduction=se_reduction) for _ in range(n_resblocks)])
        self.down2 = nn.Conv2d(n_feats*2, n_feats*4, 4, stride=2, groups=groups, padding=1)
        self.res_block3_1 = nn.Sequential(*[common.SEResBlock(conv, n_feats*4, kernel_size, groups=groups, se_reduction=se_reduction) for _ in range(n_resblocks)])
        self.down3 = nn.Conv2d(n_feats*4, n_feats*8, 4, stride=2, groups=groups, padding=1)
        self.res_block4_1 = nn.Sequential(*[common.SEResBlock(conv, n_feats*8, kernel_size, groups=groups, se_reduction=se_reduction) for _ in range(n_resblocks)])
        
        # self.lr = common.LowRankSTLayer_dilation(n_feats*8//in_channels, n_feats*2//in_channels, cube_size[0], cube_size[1], dilation=dilation, padding=padding, duration_padding=cube_size[1]//2, ranks=n_feats*2//in_channels//8)
        self.lr = common.LowRankLayer_dilation(n_feats*8, n_feats, cube_size[0], dilation=dilation, padding=padding, ranks=n_feats//16)
        
        self.res_block4_2 = nn.Sequential(*[common.SEResBlock(conv, n_feats*8, kernel_size, groups=groups, se_reduction=se_reduction) for _ in range(n_resblocks)])
        self.up3 = nn.ConvTranspose2d(n_feats*8, n_feats*4, 4, stride=2, groups=groups, padding=1)
        self.res_block3_2 = nn.Sequential(*[common.SEResBlock(conv, n_feats*4, kernel_size, groups=groups, se_reduction=se_reduction) for _ in range(n_resblocks)])
        self.up2 = nn.ConvTranspose2d(n_feats*4, n_feats*2, 4, stride=2, groups=groups, padding=1)
        self.res_block2_2 = nn.Sequential(*[common.SEResBlock(conv, n_feats*2, kernel_size, groups=groups, se_reduction=se_reduction) for _ in range(n_resblocks)])
        self.up1 = nn.ConvTranspose2d(n_feats*2, n_feats, 4, stride=2, groups=groups, padding=1)
        self.res_block1_2 = nn.Sequential(*[common.SEResBlock(conv, n_feats, kernel_size, groups=groups, se_reduction=se_reduction) for _ in range(n_resblocks)])
        
        self.tail = nn.Conv2d(n_feats, in_channels, kernel_size, groups=groups, padding=kernel_size//2)

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x0):
        batch, duration, height, width = x0.size()
        x1 = self.relu(self.head(x0))
        x2 = self.relu(self.down1(self.res_block1_1(x1)))
        x3 = self.relu(self.down2(self.res_block2_1(x2)))
        x4 = self.relu(self.down3(self.res_block3_1(x3)))
        x = self.relu(self.res_block4_1(x4))
        # x = x.view(batch, duration, self.channels, height//8, width//8).permute(0,2,1,3,4).contiguous()
        x = self.lr(x)
        # x = x.permute(0,2,1,3,4).contiguous().view(batch, duration*self.channels, height//8, width//8)
        x = self.res_block4_2(x)
        # print(x.max().cpu().numpy(), x.min().cpu().numpy())
        x = self.res_block3_2(self.up3(x+x4))
        x = self.res_block2_2(self.up2(x+x3))
        x = self.res_block1_2(self.up1(x+x2))
        x = self.tail(x+x1)

        return x

class Denoiser_LR_multi(nn.Module):
    def __init__(self, in_channels, n_feats, n_resblocks, kernel_size, cube_size=[5,3], conv=common.default_conv, se_reduction=None):
        super(Denoiser_LR_multi, self).__init__()
        groups = 1
        dilation = 3
        padding = (dilation*cube_size[0]-dilation)//2
        self.sf = 2
        # self.channels = 8*n_feats//in_channels
        self.channels = 8*n_feats

        self.head = nn.Conv2d(in_channels, n_feats, kernel_size, groups=groups, padding=kernel_size//2)

        self.res_block1_1 = nn.Sequential(*[common.SEResBlock(conv, n_feats, kernel_size, groups=groups, se_reduction=se_reduction) for _ in range(n_resblocks)])
        self.down1 = nn.Conv2d(n_feats, n_feats*2, 4, stride=2, groups=groups, padding=1)
        self.res_block2_1 = nn.Sequential(*[common.SEResBlock(conv, n_feats*2, kernel_size, groups=groups, se_reduction=se_reduction) for _ in range(n_resblocks)])
        self.down2 = nn.Conv2d(n_feats*2, n_feats*4, 4, stride=2, groups=groups, padding=1)
        self.res_block3_1 = nn.Sequential(*[common.SEResBlock(conv, n_feats*4, kernel_size, groups=groups, se_reduction=se_reduction) for _ in range(n_resblocks)])
        self.down3 = nn.Conv2d(n_feats*4, n_feats*8, 4, stride=2, groups=groups, padding=1)
        self.res_block4_1 = nn.Sequential(*[common.SEResBlock(conv, n_feats*8, kernel_size, groups=groups, se_reduction=se_reduction) for _ in range(n_resblocks)])
        
        # self.lr1 = common.LowRankLayer_dilation(n_feats, n_feats, cube_size[0], dilation=dilation, padding=padding, ranks=n_feats//16)
        self.lr2 = common.LowRankLayer_dilation(n_feats*2, n_feats, cube_size[0], dilation=dilation, padding=padding, ranks=n_feats//16)
        # self.lr3 = common.LowRankLayer_dilation(n_feats*4, n_feats, cube_size[0], dilation=dilation, padding=padding, ranks=n_feats//16)
        # self.lr4 = common.LowRankLayer_dilation(n_feats*8, n_feats, cube_size[0], dilation=dilation, padding=padding, ranks=n_feats//16)
        # self.lr = common.LowRankLayer_global(n_feats*8, n_feats, ranks=n_feats//4)
        
        # self.res_block4_2 = nn.Sequential(*[common.SEResBlock(conv, n_feats*8, kernel_size, groups=groups, se_reduction=se_reduction) for _ in range(n_resblocks)])
        self.up3 = nn.ConvTranspose2d(n_feats*8, n_feats*4, 4, stride=2, groups=groups, padding=1)
        self.res_block3_2 = nn.Sequential(*[common.SEResBlock(conv, n_feats*4, kernel_size, groups=groups, se_reduction=se_reduction) for _ in range(n_resblocks)])
        self.up2 = nn.ConvTranspose2d(n_feats*4, n_feats*2, 4, stride=2, groups=groups, padding=1)
        self.res_block2_2 = nn.Sequential(*[common.SEResBlock(conv, n_feats*2, kernel_size, groups=groups, se_reduction=se_reduction) for _ in range(n_resblocks)])
        self.up1 = nn.ConvTranspose2d(n_feats*2, n_feats, 4, stride=2, groups=groups, padding=1)
        self.res_block1_2 = nn.Sequential(*[common.SEResBlock(conv, n_feats, kernel_size, groups=groups, se_reduction=se_reduction) for _ in range(n_resblocks)])
        
        self.tail = nn.Conv2d(n_feats, in_channels, kernel_size, groups=groups, padding=kernel_size//2)

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x0):
        batch, duration, height, width = x0.size()
        x1 = self.head(x0)
        x2 = self.down1(self.res_block1_1(x1))
        x3 = self.down2(self.res_block2_1(x2))
        x4 = self.down3(self.res_block3_1(x3))
        x = self.res_block4_1(x4)
        # x = self.lr4(x)
        # x = self.res_block4_2(x)
        x = self.up3(x+x4)
        # x = self.lr3(x)
        x = self.res_block3_2(x)
        x = self.up2(x+x3)
        x = self.res_block2_2(x)
        x = self.lr2(x)
        x = self.up1(x+x2)
        # x = self.lr1(x)
        x = self.res_block1_2(x)
        x = self.tail(x+x1)

        return x

class Denoiser_Unet_EMA(nn.Module):
    def __init__(self, in_channels, n_feats, kernel_size, conv=common.default_conv, se_reduction=None):
        super(Denoiser_Unet_EMA, self).__init__()
        self.ema = common.EMAU(in_channels, n_feats)
        self.up_features = nn.Conv2d(in_channels, n_feats, kernel_size, stride=2, groups=in_channels, padding=kernel_size//2)
        self.down_features = nn.ConvTranspose2d(n_feats, in_channels, kernel_size=4, stride=2, groups=in_channels, padding=1)
        self.res_block1 = common.SEResBlock(conv, n_feats, kernel_size, groups=in_channels, se_reduction=se_reduction)
        self.res_block2 = common.SEResBlock(conv, n_feats, kernel_size, groups=in_channels, se_reduction=se_reduction)
    
    def forward(self, x):
        res = x
        x = self.up_features(x)
        x = self.res_block1(x)
        x = self.ema(x)
        x = self.res_block2(x)
        x = self.down_features(x)
        return x+res
        

def At(x, mask):
    return torch.repeat_interleave(x, repeats=mask.size(1), dim=1)*mask

def A(x, mask):
    return torch.sum(x*mask, dim=1).unsqueeze(dim=1)

def AAt(mask):
    x = torch.sum(mask*mask, dim=1).unsqueeze(dim=1)
    with torch.no_grad():
        ones = torch.div(x+1e2, x+1e2)
    return torch.where(x>0, x, ones)

class BPBlock(nn.Module):
    def __init__(self, n_resblocks, in_channels, n_feats, kernel_size, conv=common.default_conv, se_reduction=None):
        super(BPBlock, self).__init__()
        self.up_features = conv(in_channels, n_feats, kernel_size)
        self.down_features = conv(n_feats, in_channels, kernel_size)
        self.res_blocks = nn.ModuleList(
            [common.SEResBlock(conv, n_feats, kernel_size, se_reduction=se_reduction) for _ in range(n_resblocks)]
        )
        self.act = nn.ReLU(True)
    
    def forward(self, x, y, mask): 
        res = x
        x = torch.repeat_interleave(y - torch.sum(x*mask, dim=1).unsqueeze(dim=1), repeats=mask.size(1), dim=1)*mask
        x = self.up_features(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.down_features(x)
        return x+res

class ADMM_Block(nn.Module):
    def __init__(self, n_resblocks, in_channels, n_feats, kernel_size, conv=common.default_conv, se_reduction=None):
        super(ADMM_Block, self).__init__()
        self.AAt_ = None
        # self.rho = Variable(torch.tensor(1e-2), requires_grad=True)
        self.rho = nn.Parameter(torch.Tensor([0.01]))
        self.register_parameter('rho', self.rho)

        # self.denoiser = Denoiser_v1(n_resblocks, in_channels, n_feats, kernel_size, conv=conv, se_reduction=se_reduction)
        self.denoiser = Unet(in_channels, in_channels)
        # self.denoiser = Denoiser_EMA(in_channels, n_feats, kernel_size, se_reduction=se_reduction)
        # self.denoiser = Denoiser_LR(in_channels, n_feats, n_resblocks, kernel_size, se_reduction=se_reduction)

    def forward(self, v_pre, u_pre, y, mask):
        # x = v_pre + u_pre + torch.repeat_interleave((y-torch.sum((v_pre+u_pre)*mask, dim=1).unsqueeze(dim=1))/(self.rho+torch.sum(mask*mask, dim=1).unsqueeze(dim=1)+self.eps), repeats=mask.size(1), dim=1)*mask
        if self.AAt_ == None:
            self.AAt_ = AAt(mask)
        x = v_pre + u_pre + At(torch.div(y-A(v_pre+u_pre, mask), self.rho+self.AAt_), mask)
        v = self.denoiser(x - u_pre)
        u = u_pre - x + v
        return v, u

class GAP_Block(nn.Module):
    def __init__(self, n_resblocks, in_channels, n_feats, kernel_size, conv=common.default_conv, se_reduction=None):
        super(GAP_Block, self).__init__()
        # self.denoiser = Denoiser_v1(n_resblocks, in_channels, n_feats, kernel_size, conv=conv, se_reduction=se_reduction)
        self.denoiser = Unet(in_channels, in_channels)
        # self.denoiser = Denoiser_EMA(in_channels, n_feats, kernel_size, se_reduction=se_reduction)

    def forward(self, v_pre, y, mask):
        # x = v_pre + torch.repeat_interleave((y-torch.sum(v_pre*mask, dim=1).unsqueeze(dim=1))/(torch.sum(mask*mask, dim=1).unsqueeze(dim=1)+self.eps), repeats=mask.size(1), dim=1)*mask
        x = v_pre + At(torch.div(y-A(v_pre, mask), AAt(mask)), mask)
        v = self.denoiser(x)
        return v

class MSEN(nn.Module):
    def __init__(self, n_resblocks, n_memblocks, in_channels, out_channels, n_feats, stage=1, conv=common.default_conv, se_reduction=None):
        super(MSEN, self).__init__()

        kernel_size = 3 

        # define head module
        self.head = conv(in_channels, n_feats, kernel_size)  

        # define body module
        self.body = nn.ModuleList(
            # [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction, if_last=(i==n_memblocks-1), out_channels=out_channels) for i in range(n_memblocks)]
            [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction) for i in range(n_memblocks)]
        )

        # define tail module
        self.tail = conv(n_feats, out_channels, kernel_size)

        # self.stage = stage
        # self.head = []
        # self.body = []
        # self.tail = []

        # for _ in range(int(np.exp2(stage-1))):
        #     # define head module
        #     self.head.append(conv(in_channels, n_feats, kernel_size))   

        #     # define body module
        #     self.body.append(nn.ModuleList(
        #         # [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction, if_last=(i==n_memblocks-1), out_channels=out_channels) for i in range(n_memblocks)]
        #         [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction) for i in range(n_memblocks)]
        #     ))

        #     # define tail module
        #     self.tail.append(conv(n_feats, out_channels, kernel_size))        

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        x = torch.cat([torch.repeat_interleave(x, repeats=mask.size(1), dim=1)*mask, x], dim=1)
        x = self.head(x)
        x = self.relu(x)

        ys = [x]
        for memory_block in self.body:
            x = memory_block(x, ys)

        x = self.tail(x)
        return x

        # out = []
        # for i in range(int(np.exp2(self.stage-1))):
        #     y = torch.cat([torch.repeat_interleave(x[:,i:i+1,:,:], repeats=mask.size(1)//self.stage, dim=1)*mask[:,2*i:2*i+2], x[:,i:i+1,:,:]], dim=1)
        #     y = self.head[i](y)
        #     y = self.relu(y)

        #     ys = [y]
        #     for memory_block in self.body[i]:
        #         y = memory_block(y, ys)
            
        #     y = self.tail[i](y)

        #     out.append(y)
        # return torch.cat(out, dim=1) 


    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))    

class MSEN1(nn.Module):
    def __init__(self, n_resblocks, n_memblocks, in_channels, out_channels, n_feats, stage=1, conv=common.default_conv, se_reduction=None):
        super(MSEN1, self).__init__()

        kernel_size = 3 

        # define head module
        self.head = conv(in_channels, n_feats, kernel_size)  

        # define body module
        self.body = nn.ModuleList(
            # [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction, if_last=(i==n_memblocks-1), out_channels=out_channels) for i in range(n_memblocks)]
            [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction) for i in range(n_memblocks)]
        )
        # self.ema = nn.ModuleList(
        #     [common.EMAU(2, n_feats, k=64) for i in range(n_memblocks)]
        # )

        # define tail module
        self.tail = conv(n_feats, out_channels, kernel_size)      

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        x = torch.cat([torch.repeat_interleave(x, repeats=mask.size(1), dim=1)*mask, x], dim=1)
        x = self.head(x)
        x = self.relu(x)

        ys = [x]
        for memory_block in self.body:
            x = memory_block(x, ys)
        # for memory_block, ema in zip(self.body, self.ema):
        #     x = memory_block(x, ys)
        #     x = ema(x)

        x = self.tail(x)
        x = self.relu(x)
        return x

class ADMMN(nn.Module):
    def __init__(self, n_resblocks, n_admmblocks, in_channels, n_feats, conv=common.default_conv, se_reduction=None):
        super(ADMMN, self).__init__()

        kernel_size = 3
        self.admm_blocks = nn.ModuleList(
            [ADMM_Block(n_resblocks, in_channels, n_feats, kernel_size, se_reduction=se_reduction) for _ in range(n_admmblocks)]
        )

        # self.denoiser = Denoiser_v1(n_resblocks, in_channels, n_feats, kernel_size, conv=conv, se_reduction=se_reduction)
        self.denoiser = Unet(in_channels, in_channels)
        # self.denoiser = Denoiser_EMA(in_channels, n_feats, kernel_size, se_reduction=se_reduction)
        # self.denoiser = Denoiser_LR(in_channels, n_feats, n_resblocks, kernel_size, se_reduction=se_reduction)
        # self.rho = Variable(torch.tensor(1e-2), requires_grad=True)
        self.rho = nn.Parameter(torch.Tensor([0.01]))
        self.register_parameter('rho', self.rho)
        self.AAt_ = None

    def forward(self, y, mask):
        # v = torch.repeat_interleave(y, repeats=mask.size(1), dim=1)*mask
        # x = v + torch.repeat_interleave((y-torch.sum(v*mask, dim=1).unsqueeze(dim=1))/(self.rho+torch.sum(mask*mask, dim=1).unsqueeze(dim=1)+self.eps), repeats=mask.size(1), dim=1)*mask
        if self.AAt_ == None:
            self.AAt_ = AAt(mask)
        v = At(y, mask)
        x = v + At(torch.div(y-A(v, mask), self.rho+self.AAt_), mask)
        v = self.denoiser(x)
        u = v - x

        v_list = [v]
        v_list.append(v)
        for block in self.admm_blocks:
            v, u = block(v, u, y, mask)
            v_list.append(v)
        
        return v_list

class GAPN1(nn.Module):
    def __init__(self, n_resblocks, n_gapblocks, in_channels, n_feats, conv=common.default_conv, se_reduction=None):
        super(GAPN1, self).__init__()

        kernel_size = 3
        self.gap_blocks = nn.ModuleList(
            [GAP_Block(n_resblocks, in_channels, n_feats, kernel_size, se_reduction=se_reduction) for _ in range(n_gapblocks)]
        )

        # self.denoiser = Denoiser_v1(n_resblocks, in_channels, n_feats, kernel_size, conv=conv, se_reduction=se_reduction)
        self.denoiser = Unet(in_channels, in_channels)
        # self.denoiser = Denoiser_EMA(in_channels, n_feats, kernel_size, se_reduction=se_reduction)

    def forward(self, y, mask):
        v = torch.repeat_interleave(y, repeats=mask.size(1), dim=1)*mask
        # x = v + torch.repeat_interleave((y-torch.sum(v*mask, dim=1).unsqueeze(dim=1))/(torch.sum(mask*mask, dim=1).unsqueeze(dim=1)+self.eps), repeats=mask.size(1), dim=1)*mask
        x = v + At(torch.div(y-A(v, mask), AAt(mask)), mask)
        v = self.denoiser(x)

        v_list = []
        v_list.append(v)
        for block in self.gap_blocks:
            v = block(v, y, mask)
            v_list.append(v)
        
        return v_list 

class GAPN2(nn.Module):
    def __init__(self, n_resblocks, n_gapblocks, in_channels, n_feats, conv=common.default_conv, se_reduction=None):
        super(GAPN2, self).__init__()

        kernel_size = 3
        self.gap_blocks1 = nn.ModuleList(
            [GAP_Block(n_resblocks, in_channels, n_feats, kernel_size, se_reduction=se_reduction) for _ in range(n_gapblocks)]
        )
        self.gap_blocks2 = nn.ModuleList(
            [GAP_Block(n_resblocks, in_channels, n_feats, kernel_size, se_reduction=se_reduction) for _ in range(n_gapblocks)]
        )

        # self.denoiser = Denoiser_v1(n_resblocks, in_channels, n_feats, kernel_size, conv=conv, se_reduction=se_reduction)
        self.denoiser1 = Unet(in_channels, in_channels)
        self.denoiser2 = Unet(in_channels, in_channels)

    def forward(self, y, mask):
        y1 = y[:,:1,:,:]
        mask1 = mask[:,:2,:,:]
        v1 = torch.repeat_interleave(y1, repeats=mask1.size(1), dim=1)*mask1
        x1 = v1 + At(torch.div(y1-A(v1, mask1), AAt(mask1)), mask1)
        v1 = self.denoiser1(x1)

        y2 = y[:,1:,:,:]
        mask2 = mask[:,2:,:,:]
        v2 = torch.repeat_interleave(y2, repeats=mask2.size(1), dim=1)*mask2
        x2 = v2 + At(torch.div(y2-A(v2, mask2), AAt(mask2)), mask2)
        v2 = self.denoiser2(x2)

        for i in range(len(self.gap_blocks1)):
            v1 = self.gap_blocks1[i](v1, y1, mask1)
            v2 = self.gap_blocks2[i](v2, y2, mask2)
        
        return torch.cat([v1,v2], dim=1) 

class GAPN4(nn.Module):
    def __init__(self, n_resblocks, n_gapblocks, in_channels, n_feats, conv=common.default_conv, se_reduction=None):
        super(GAPN4, self).__init__()

        kernel_size = 3
        self.gap_blocks1 = nn.ModuleList(
            [GAP_Block(n_resblocks, in_channels, n_feats, kernel_size, se_reduction=se_reduction) for _ in range(n_gapblocks)]
        )
        self.gap_blocks2 = nn.ModuleList(
            [GAP_Block(n_resblocks, in_channels, n_feats, kernel_size, se_reduction=se_reduction) for _ in range(n_gapblocks)]
        )
        self.gap_blocks3 = nn.ModuleList(
            [GAP_Block(n_resblocks, in_channels, n_feats, kernel_size, se_reduction=se_reduction) for _ in range(n_gapblocks)]
        )
        self.gap_blocks4 = nn.ModuleList(
            [GAP_Block(n_resblocks, in_channels, n_feats, kernel_size, se_reduction=se_reduction) for _ in range(n_gapblocks)]
        )

        # self.denoiser = Denoiser_v1(n_resblocks, in_channels, n_feats, kernel_size, conv=conv, se_reduction=se_reduction)
        self.denoiser1 = Unet(in_channels, in_channels)
        self.denoiser2 = Unet(in_channels, in_channels)
        self.denoiser3 = Unet(in_channels, in_channels)
        self.denoiser4 = Unet(in_channels, in_channels)

    def forward(self, y, mask):
        y1 = y[:,:1,:,:]
        mask1 = mask[:,:2,:,:]
        v1 = torch.repeat_interleave(y1, repeats=mask1.size(1), dim=1)*mask1
        x1 = v1 + At(torch.div(y1-A(v1, mask1), AAt(mask1)), mask1)
        v1 = self.denoiser1(x1)

        y2 = y[:,1:2,:,:]
        mask2 = mask[:,2:4,:,:]
        v2 = torch.repeat_interleave(y2, repeats=mask2.size(1), dim=1)*mask2
        x2 = v2 + At(torch.div(y2-A(v2, mask2), AAt(mask2)), mask2)
        v2 = self.denoiser2(x2)

        y3 = y[:,2:3,:,:]
        mask3 = mask[:,4:6,:,:]
        v3 = torch.repeat_interleave(y3, repeats=mask3.size(1), dim=1)*mask3
        x3 = v3 + At(torch.div(y3-A(v3, mask3), AAt(mask3)), mask3)
        v3 = self.denoiser3(x3)

        y4 = y[:,3:,:,:]
        mask4 = mask[:,6:8,:,:]
        v4 = torch.repeat_interleave(y4, repeats=mask4.size(1), dim=1)*mask4
        x4 = v4 + At(torch.div(y4-A(v4, mask4), AAt(mask4)), mask4)
        v4 = self.denoiser4(x4)

        for i in range(len(self.gap_blocks1)):
            v1 = self.gap_blocks1[i](v1, y1, mask1)
            v2 = self.gap_blocks2[i](v2, y2, mask2)
            v3 = self.gap_blocks3[i](v3, y3, mask3)
            v4 = self.gap_blocks4[i](v4, y4, mask4)
        
        return torch.cat([v1,v2,v3,v4], dim=1) 

class BPN1(nn.Module):
    def __init__(self, n_resblocks, n_bpblocks, in_channels, n_feats, conv=common.default_conv, se_reduction=None):
        super(BPN1, self).__init__()

        kernel_size = 3

        # self.up_features = conv(in_channels, n_feats, kernel_size)
        # self.down_features = conv(n_feats, in_channels, kernel_size)
        # self.res_blocks = nn.ModuleList(
        #     [common.SEResBlock(conv, n_feats, kernel_size, se_reduction=se_reduction) for _ in range(n_resblocks)]
        # )

        self.body = nn.ModuleList(
            [BPBlock(n_resblocks, in_channels, n_feats, kernel_size, se_reduction=se_reduction) for _ in range(n_bpblocks)]
        )
    
    def forward(self, y, mask):
        # x = torch.repeat_interleave(y, repeats=mask.size(1), dim=1)*mask
        # res = x
        # x = self.up_features(x)
        # for block in self.res_blocks:
        #     x = block(x)
        # x = self.down_features(x)
        # x = res+x

        x = torch.repeat_interleave(y/torch.sum(mask*mask, dim=1).unsqueeze(dim=1), repeats=mask.size(1), dim=1)

        for block in self.body:
            x = block(x, y, mask)

        return x
    

class MSEN2(nn.Module):
    def __init__(self, n_resblocks, n_memblocks, in_channels, out_channels, n_feats, stage=1, conv=common.default_conv, se_reduction=None):
        super(MSEN2, self).__init__()

        kernel_size = 3 

        # define head module
        self.head1 = conv(in_channels, n_feats, kernel_size)  
        self.head2 = conv(in_channels, n_feats, kernel_size)  

        # define body module
        self.body1 = nn.ModuleList(
            # [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction, if_last=(i==n_memblocks-1), out_channels=out_channels) for i in range(n_memblocks)]
            [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction) for i in range(n_memblocks)]
        )
        self.body2 = nn.ModuleList(
            # [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction, if_last=(i==n_memblocks-1), out_channels=out_channels) for i in range(n_memblocks)]
            [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction) for i in range(n_memblocks)]
        )

        # define tail module
        self.tail1 = conv(n_feats, out_channels, kernel_size)      
        self.tail2 = conv(n_feats, out_channels, kernel_size)      

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        x1 = torch.cat([torch.repeat_interleave(x[:,:1,:,:], repeats=mask.size(1)//2, dim=1)*mask[:,:2,:,:], x[:,:1,:,:]], dim=1)
        x1 = self.head1(x1)
        x1 = self.relu(x1)

        x2 = torch.cat([torch.repeat_interleave(x[:,1:,:,:], repeats=mask.size(1)//2, dim=1)*mask[:,2:,:,:], x[:,1:,:,:]], dim=1)
        x2 = self.head2(x2)
        x2 = self.relu(x2)

        ys1 = [x1]
        ys2 = [x2]
        for i in range(len(self.body1)):
            x1 = self.body1[i](x1, ys1)
            x2 = self.body2[i](x2, ys2)

        x1 = self.tail1(x1)
        x2 = self.tail2(x2)
        return torch.cat([x1,x2], dim=1)

class MSEN4(nn.Module):
    def __init__(self, n_resblocks, n_memblocks, in_channels, out_channels, n_feats, stage=1, conv=common.default_conv, se_reduction=None):
        super(MSEN4, self).__init__()

        kernel_size = 3 

        # define head module
        self.head1 = conv(in_channels, n_feats, kernel_size)  
        self.head2 = conv(in_channels, n_feats, kernel_size)  
        self.head3 = conv(in_channels, n_feats, kernel_size)  
        self.head4 = conv(in_channels, n_feats, kernel_size)  

        # define body module
        self.body1 = nn.ModuleList(
            # [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction, if_last=(i==n_memblocks-1), out_channels=out_channels) for i in range(n_memblocks)]
            [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction) for i in range(n_memblocks)]
        )
        self.body2 = nn.ModuleList(
            # [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction, if_last=(i==n_memblocks-1), out_channels=out_channels) for i in range(n_memblocks)]
            [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction) for i in range(n_memblocks)]
        )
        self.body3 = nn.ModuleList(
            # [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction, if_last=(i==n_memblocks-1), out_channels=out_channels) for i in range(n_memblocks)]
            [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction) for i in range(n_memblocks)]
        )
        self.body4 = nn.ModuleList(
            # [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction, if_last=(i==n_memblocks-1), out_channels=out_channels) for i in range(n_memblocks)]
            [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction) for i in range(n_memblocks)]
        )

        # define tail module
        self.tail1 = conv(n_feats, out_channels, kernel_size)      
        self.tail2 = conv(n_feats, out_channels, kernel_size)      
        self.tail3 = conv(n_feats, out_channels, kernel_size)      
        self.tail4 = conv(n_feats, out_channels, kernel_size)      

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        x1 = torch.cat([torch.repeat_interleave(x[:,:1,:,:], repeats=mask.size(1)//4, dim=1)*mask[:,:2,:,:], x[:,:1,:,:]], dim=1)
        x1 = self.head1(x1)
        x1 = self.relu(x1)

        x2 = torch.cat([torch.repeat_interleave(x[:,1:2,:,:], repeats=mask.size(1)//4, dim=1)*mask[:,2:4,:,:], x[:,1:2,:,:]], dim=1)
        x2 = self.head2(x2)
        x2 = self.relu(x2)

        x3 = torch.cat([torch.repeat_interleave(x[:,2:3,:,:], repeats=mask.size(1)//4, dim=1)*mask[:,4:6,:,:], x[:,2:3,:,:]], dim=1)
        x3 = self.head2(x3)
        x3 = self.relu(x3)

        x4 = torch.cat([torch.repeat_interleave(x[:,3:,:,:], repeats=mask.size(1)//4, dim=1)*mask[:,6:,:,:], x[:,3:,:,:]], dim=1)
        x4 = self.head2(x4)
        x4 = self.relu(x4)

        ys1 = [x1]
        ys2 = [x2]
        ys3 = [x3]
        ys4 = [x4]
        for i in range(len(self.body1)):
            x1 = self.body1[i](x1, ys1)
            x2 = self.body2[i](x2, ys2)
            x3 = self.body3[i](x3, ys3)
            x4 = self.body4[i](x4, ys4)

        x1 = self.tail1(x1)
        x2 = self.tail2(x2)
        x3 = self.tail3(x3)
        x4 = self.tail4(x4)
        return torch.cat([x1,x2,x3,x4], dim=1)

class MSEN8(nn.Module):
    def __init__(self, n_resblocks, n_memblocks, in_channels, out_channels, n_feats, conv=common.default_conv, se_reduction=None):
        super(MSEN8, self).__init__()

        kernel_size = 3 

        # define head module
        self.head1 = conv(in_channels, n_feats, kernel_size)  
        self.head2 = conv(in_channels, n_feats, kernel_size)  
        self.head3 = conv(in_channels, n_feats, kernel_size)  
        self.head4 = conv(in_channels, n_feats, kernel_size)  
        self.head5 = conv(in_channels, n_feats, kernel_size)  
        self.head6 = conv(in_channels, n_feats, kernel_size)  
        self.head7 = conv(in_channels, n_feats, kernel_size)  
        self.head8 = conv(in_channels, n_feats, kernel_size)  

        # define body module
        self.body1 = nn.ModuleList(
            [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction) for i in range(n_memblocks)]
        )
        self.body2 = nn.ModuleList(
            [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction) for i in range(n_memblocks)]
        )
        self.body3 = nn.ModuleList(
            [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction) for i in range(n_memblocks)]
        )
        self.body4 = nn.ModuleList(
            [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction) for i in range(n_memblocks)]
        )
        self.body5 = nn.ModuleList(
            [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction) for i in range(n_memblocks)]
        )
        self.body6 = nn.ModuleList(
            [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction) for i in range(n_memblocks)]
        )
        self.body7 = nn.ModuleList(
            [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction) for i in range(n_memblocks)]
        )
        self.body8 = nn.ModuleList(
            [MemoryBlock(n_resblocks, i+1, n_feats, kernel_size, se_reduction=se_reduction) for i in range(n_memblocks)]
        )

        # define tail module
        self.tail1 = conv(n_feats, out_channels, kernel_size)
        self.tail2 = conv(n_feats, out_channels, kernel_size)
        self.tail3 = conv(n_feats, out_channels, kernel_size)
        self.tail4 = conv(n_feats, out_channels, kernel_size)  
        self.tail5 = conv(n_feats, out_channels, kernel_size)  
        self.tail6 = conv(n_feats, out_channels, kernel_size)  
        self.tail7 = conv(n_feats, out_channels, kernel_size)  
        self.tail8 = conv(n_feats, out_channels, kernel_size)  

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        x_A1 = At(x, mask)
        x_A2 = torch.div(x_A1, AAt(mask))
        x1 = torch.cat([x_A1[:,:1,:,:], x_A2[:,:1,:,:], x], dim=1)
        x1 = self.head1(x1)
        x1 = self.relu(x1)

        x2 = torch.cat([x_A1[:,1:2,:,:], x_A2[:,1:2,:,:], x], dim=1)
        x2 = self.head2(x2)
        x2 = self.relu(x2)

        x3 = torch.cat([x_A1[:,2:3,:,:], x_A2[:,2:3,:,:], x], dim=1)
        x3 = self.head2(x3)
        x3 = self.relu(x3)

        x4 = torch.cat([x_A1[:,3:4,:,:], x_A2[:,3:4,:,:], x], dim=1)
        x4 = self.head2(x4)
        x4 = self.relu(x4)

        x5 = torch.cat([x_A1[:,4:5,:,:], x_A2[:,4:5,:,:], x], dim=1)
        x5 = self.head2(x5)
        x5 = self.relu(x5)

        x6 = torch.cat([x_A1[:,5:6,:,:], x_A2[:,5:6,:,:], x], dim=1)
        x6 = self.head2(x6)
        x6 = self.relu(x6)

        x7 = torch.cat([x_A1[:,6:7,:,:], x_A2[:,6:7,:,:], x], dim=1)
        x7 = self.head2(x7)
        x7 = self.relu(x7)

        x8 = torch.cat([x_A1[:,7:,:,:], x_A2[:,7:,:,:], x], dim=1)
        x8 = self.head2(x8)
        x8 = self.relu(x8)

        ys1 = [x1]
        ys2 = [x2]
        ys3 = [x3]
        ys4 = [x4]
        ys5 = [x5]
        ys6 = [x6]
        ys7 = [x7]
        ys8 = [x8]
        for i in range(len(self.body1)):
            x1 = self.body1[i](x1, ys1)
            x2 = self.body2[i](x2, ys2)
            x3 = self.body3[i](x3, ys3)
            x4 = self.body4[i](x4, ys4)
            x5 = self.body5[i](x5, ys5)
            x6 = self.body6[i](x6, ys6)
            x7 = self.body7[i](x7, ys7)
            x8 = self.body8[i](x8, ys8)

        x1 = self.tail1(x1)
        x2 = self.tail2(x2)
        x3 = self.tail3(x3)
        x4 = self.tail4(x4)
        x5 = self.tail5(x5)
        x6 = self.tail6(x6)
        x7 = self.tail7(x7)
        x8 = self.tail8(x8)
        return torch.cat([x1,x2,x3,x4,x5,x6,x7,x8], dim=1)

class Discriminator(nn.Module):
    def __init__(self, frames, n_feats):
        super(Discriminator, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(frames, n_feats, kernel_size=4, padding=1, stride=2), # 16x128x128
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_feats, 2*n_feats, kernel_size=4, padding=1, stride=2), # 32x64x64
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(2*n_feats, 4*n_feats, kernel_size=4, padding=1, stride=2), # 64x32x32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(4*n_feats, 8*n_feats, kernel_size=4, padding=1, stride=2), # 128x16x16
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(8*n_feats, 16*n_feats, kernel_size=4, padding=1, stride=2), # 256x8x8
            nn.LeakyReLU(0.2, inplace=True)
        )

        # self.classifer = nn.Linear(16*n_feats*8*8, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)
        # x = x.view(x.size(0), -1)
        # x = self.sigmoid(self.classifer(x))

        return x

def Adversarial(disc, disc_optim, bce, fake, real):
    fake_detach = fake.detach()
    disc_optim.zero_grad()
    d_fake = disc(fake_detach)
    d_real = disc(real)
    label_fake = torch.zeros_like(d_fake)
    label_real = torch.ones_like(d_real)
    loss_d_fake = bce(d_fake, label_fake) 
    loss_d_real = bce(d_real, label_real)
    loss_d = loss_d_fake + loss_d_real
    loss_d.backward()
    disc_optim.step()

    d_fake_bp = disc(fake)
    label_real = torch.ones_like(d_fake_bp)
    loss_g = bce(d_fake_bp, label_real)

    return loss_g, loss_d_real, loss_d_fake

if __name__ == '__main__':
    y = torch.cuda.FloatTensor(1, 1, 256, 256).fill_(0)
    mask = torch.cuda.FloatTensor(1, 2, 256, 256).fill_(0)
    n_feats = 32
    # net1 = MSEN1(n_resblocks=4, n_memblocks=4, in_channels=2+1, out_channels=2, n_feats=n_feats, se_reduction=8).cuda()
    net1 = GAPN1(n_resblocks=4, n_gapblocks=4, in_channels=2, n_feats=n_feats, se_reduction=8).cuda()
    net2 = MSEN2(n_resblocks=4, n_memblocks=4, in_channels=2+1, out_channels=2, n_feats=n_feats, se_reduction=8).cuda()
    net3 = MSEN4(n_resblocks=4, n_memblocks=4, in_channels=2+1, out_channels=2, n_feats=n_feats, se_reduction=8).cuda()
    # net2_1 = MSEN(n_resblocks=4, n_memblocks=4, in_channels=2+1, out_channels=2, n_feats=n_feats, se_reduction=8).cuda()
    # net2_2 = MSEN(n_resblocks=4, n_memblocks=4, in_channels=2+1, out_channels=2, n_feats=n_feats, se_reduction=8).cuda()
    # net3_1 = MSEN(n_resblocks=4, n_memblocks=4, in_channels=2+1, out_channels=2, n_feats=n_feats, se_reduction=8).cuda()
    # net3_2 = MSEN(n_resblocks=4, n_memblocks=4, in_channels=2+1, out_channels=2, n_feats=n_feats, se_reduction=8).cuda()
    # net3_3 = MSEN(n_resblocks=4, n_memblocks=4, in_channels=2+1, out_channels=2, n_feats=n_feats, se_reduction=8).cuda()
    # net3_4 = MSEN(n_resblocks=4, n_memblocks=4, in_channels=2+1, out_channels=2, n_feats=n_feats, se_reduction=8).cuda()
    out1 = net1(y, mask)
    out2 = net2(out1, torch.cat([mask, mask], dim=1))
    out3 = net3(out2, torch.cat([mask, mask, mask, mask], dim=1))
    print(out1[:,0,:,:].size())
    # out2_1 = net2_1(out1[:,:1,:,:], mask)
    # out2_2 = net2_2(out1[:,1:,:,:], mask)
    # out3_1 = net3_1(out2_1[:,:1,:,:], mask)
    # out3_2 = net3_2(out2_1[:,1:,:,:], mask)
    # out3_3 = net3_3(out2_2[:,:1,:,:], mask)
    # out3_4 = net3_4(out2_2[:,1:,:,:], mask)
    import ipdb; ipdb.set_trace()
