import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.parameter import Parameter


def default_conv(in_channels, out_channels, kernel_size, groups=1, bias=False):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, groups=groups,
        padding=(kernel_size//2), bias=bias)

class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.d_conv(x)
        return x

class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, groups=1,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, groups=groups, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class SEResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, groups=1,
        bias=True, bn=False, In=False, act=nn.ReLU(True), res_scale=1, se_reduction=None, bn_feat=None):

        super(SEResBlock, self).__init__()
        m = []
        bn_feat = bn_feat or n_feat

        m.append(conv(n_feat, bn_feat, kernel_size, groups=groups, bias=bias))
        if bn: m.append(nn.BatchNorm2d(bn_feat))
        elif In: m.append(nn.InstanceNorm2d(bn_feat, affine=True, track_running_stats=True))
        m.append(act)
        m.append(conv(bn_feat, n_feat, kernel_size, groups=groups, bias=bias))
        if bn: m.append(nn.BatchNorm2d(n_feat))
        elif In: m.append(nn.InstanceNorm2d(n_feat, affine=True, track_running_stats=True))

        if se_reduction is not None:
            m.append(SElayer(n_feat, se_reduction))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class SEpreResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1, se_reduction=None, bn_feat=None):

        super(SEpreResBlock, self).__init__()
        m = []
        bn_feat = bn_feat or n_feat

        if bn: m.append(nn.BatchNorm2d(n_feat))
        m.append(act)
        m.append(conv(n_feat, bn_feat, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm2d(bn_feat))
        m.append(act)
        m.append(conv(bn_feat, n_feat, kernel_size, bias=bias))

        if se_reduction is not None:
            m.append(SElayer(n_feat, se_reduction))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class SElayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SElayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class LowRankLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, ranks=3):
        super(LowRankLayer, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # self.k_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.q_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.v_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.U_w = nn.Parameter(torch.randn(kernel_size**2, ranks), requires_grad=True)
        self.V_w = nn.Parameter(torch.randn(out_channels, ranks), requires_grad=True)

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding], mode='reflect')
        q_out = self.q_conv(x)
        # k_out = self.k_conv(padded_x)
        v_out = self.v_conv(padded_x)

        # k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        # k_out = k_out.contiguous().view(batch, self.out_channels, height, width, -1)

        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.contiguous().view(batch, self.out_channels, height, width, -1)

        q_out = q_out.view(batch, self.out_channels, height, width, 1)

        # out = torch.einsum('bchwk,bchwq -> bhwkq', k_out, q_out).view(batch, height, width, -1)
        # out = F.softmax(out, dim=-1)
        # out = F.relu(out - 1.0/self.kernel_size**2, inplace=True)
        # out = torch.div(out, out+1e-9)

        # v_out_selected = v_out * out.view(batch, 1, height, width, -1)
        U = torch.einsum('bchwk,kr -> bchwr', v_out, self.U_w)
        V = torch.einsum('bchwk,cr -> brhwk', q_out, self.V_w)
        UV = torch.einsum('bchwr,brhwk -> bchwk', U, V)

        return UV.view(batch, self.out_channels, height, width)

# class LowRankLayer_dilation(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, bias=False, ranks=3, stage_num=1):
#         super(LowRankLayer_dilation, self).__init__()
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.stage_num = stage_num
#         self.ranks = ranks

#         self.eps_U = None
#         self.eps_V = None

#         self.head_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
#         self.tail_conv = nn.Conv2d(out_channels, in_channels, kernel_size=1, bias=bias)

#         self.unfold = nn.Unfold(kernel_size=[kernel_size, kernel_size], dilation=[dilation, dilation], padding=0, stride=stride)

#         # self.U_w = nn.Parameter(torch.randn(kernel_size**2, ranks), requires_grad=True)
#         # # self.U_w2 = nn.Parameter(torch.randn(ranks, ranks), requires_grad=True)
#         # self.V_w = nn.Parameter(torch.randn(out_channels, ranks), requires_grad=True)
#         # # self.V_w2 = nn.Parameter(torch.randn(ranks, ranks), requires_grad=True)
        
#         # init.normal_(self.U_w, 0, math.sqrt(2. / ranks))
#         # # init.normal_(self.U_w2, 0, math.sqrt(2. / ranks))
#         # init.normal_(self.V_w, 0, math.sqrt(2. / ranks))
#         # # init.normal_(self.V_w2, 0, math.sqrt(2. / ranks))

#     def forward(self, x, eps=1e-6):
#         batch, channels, height, width = x.size()
#         x = self.head_conv(x)
#         # x = F.sigmoid(x)
#         if torch.max(x).cpu().detach().numpy() < 1e-16:
#             print(torch.max(x).cpu().detach().numpy())
#         x = F.relu(x, inplace=True)
#         padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding], mode='replicate')
#         # X = x.view(batch, self.out_channels, height, width, 1)
#         X = self.unfold(padded_x)
#         X = X.view(batch, self.out_channels, -1, height, width).permute(0,1,3,4,2).contiguous()

#         # U = self.unfold(padded_x)
#         # U = U.view(batch, self.out_channels, height, width, -1)

#         # V = x.view(batch, self.out_channels, height, width, 1)

#         # U = torch.einsum('bchwk,kr -> bchwr', U, self.U_w)
#         # # U = F.relu(U, inplace=True)
#         # # U = torch.einsum('bchwk,kr -> bchwr', U, self.U_w2)
#         # # U = self._l2norm(U, dim=-1)
#         # V = torch.einsum('bchwk,cr -> brhwk', V, self.V_w)
#         # # V = F.relu(V, inplace=True)
#         # # V = torch.einsum('bchwk,cr -> brhwk', V, self.V_w2)

#         # UV = torch.einsum('bchwr,brhwk -> bchwk', U, V)
#         # UV = UV[:,:,:,:,0]

#         U = X.mean(dim=-1, keepdim=True).repeat(1,1,1,1,self.ranks)
#         V = X.mean(dim=1, keepdim=True).repeat(1,self.ranks,1,1,1)
#         with torch.no_grad():
#             # UV = torch.einsum('bchwr,brhwk -> bchwk', U, V)[:,:,:,:,self.kernel_size//2*self.kernel_size + self.kernel_size//2]
#             # print(torch.mean((x-UV)*(x-UV)).cpu().detach().numpy())
#             # import cv2
#             # import numpy as np
#             # for i in range(x.size()[1]):
#             #     cv2.imshow('pre', x.cpu().numpy()[0,i,:,:]/np.amax(x.cpu().numpy()[0,i,:,:]))
#             #     cv2.imshow('post', UV.cpu().numpy()[0,i,:,:]/np.amax(UV.cpu().numpy()[0,i,:,:]))
#             #     cv2.imshow('X', X.cpu().numpy()[0,i,:,:,1]/np.amax(X.cpu().numpy()[0,i,:,:,1]))
#             #     cv2.waitKey()

#             # if self.eps_U == None:
#             #     self.eps_U = torch.ones_like(U)*eps
#             # if self.eps_V == None:
#             #     self.eps_V = torch.ones_like(V)*eps
#             for _ in range(self.stage_num-1):
#                 numerator = torch.einsum('bchwr,bchwk -> brhwk', U, X)
#                 denominator = torch.einsum('bchwr,bchwk -> brhwk', U, torch.einsum('bchwr,brhwk -> bchwk', U, V))
#                 V = V * torch.div(numerator, denominator+eps)
#                 # V = torch.where(V>self.eps_V, V, self.eps_V)

#                 numerator = torch.einsum('bchwk,brhwk -> bchwr', X, V)
#                 denominator = torch.einsum('bchwk,brhwk -> bchwr', torch.einsum('bchwr,brhwk -> bchwk', U, V), V)
#                 U = U * torch.div(numerator, denominator+eps)
#                 # U = torch.where(U>self.eps_U, U, self.eps_U)

#                 UV = torch.einsum('bchwr,brhwk -> bchwk', U, V)[:,:,:,:,self.kernel_size//2*self.kernel_size + self.kernel_size//2]
#                 print(torch.mean((x-UV)*(x-UV)).cpu().detach().numpy())

#         numerator = torch.einsum('bchwr,bchwk -> brhwk', U, X)
#         denominator = torch.einsum('bchwr,bchwk -> brhwk', U, torch.einsum('bchwr,brhwk -> bchwk', U, V))
#         V = V * torch.div(numerator, denominator+eps)
#         # V = torch.where(V>self.eps_V, V, self.eps_V)

#         numerator = torch.einsum('bchwk,brhwk -> bchwr', X, V)
#         denominator = torch.einsum('bchwk,brhwk -> bchwr', torch.einsum('bchwr,brhwk -> bchwk', U, V), V)
#         U = U * torch.div(numerator, denominator+eps)
#         # U = torch.where(U>self.eps_U, U, self.eps_U)
#         # UV = torch.einsum('bchwr,brhwk -> bchwk', U, V)[:,:,:,:,self.kernel_size//2*self.kernel_size + self.kernel_size//2]
#         # print(torch.mean((x-UV)*(x-UV)).cpu().detach().numpy())
#         # print('U', torch.max(U), torch.min(U))
#         # print('V', torch.max(V), torch.min(V))

#         UV = torch.einsum('bchwr,brhwk -> bchwk', U, V)
#         # UV = UV[:,:,:,:,:,0]
#         UV = UV[:,:,:,:,self.kernel_size//2*self.kernel_size + self.kernel_size//2]
#         # UV = UV.view(batch, self.out_channels, duration, height, width, self.kernel_size, self.kernel_size, self.duration_kernel_size).contiguous()
#         # UV = UV[:,:,:,:,:,self.kernel_size//2,self.kernel_size//2,self.duration_kernel_size//2]
#         # UV = UV.view(batch, self.out_channels, duration, height, width).contiguous()
#         # print(torch.max(UV).cpu().detach().numpy())

#         out = self.tail_conv(UV)#+res
#         out = F.relu(out, inplace=True)
#         # print(torch.max(out).cpu().detach().numpy())
#         # import cv2
#         # import numpy as np
#         # for i in range(x.size()[1]):
#         #     cv2.imshow('pre', x.cpu().numpy()[0,i,:,:]/np.amax(x.cpu().numpy()[0,i,:,:]))
#         #     cv2.imshow('post', UV.cpu().numpy()[0,i,:,:]/np.amax(UV.cpu().numpy()[0,i,:,:]))
#         #     cv2.waitKey()

#         return out.contiguous()

class LowRankLayer_dilation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, bias=False, ranks=3, stage_num=1):
        super(LowRankLayer_dilation, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.stage_num = stage_num
        self.ranks = ranks

        self.eps_U = None
        self.eps_V = None

        self.head_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.tail_conv = nn.Conv2d(out_channels, in_channels, kernel_size=1, bias=bias)

        self.unfold = nn.Unfold(kernel_size=[kernel_size, kernel_size], dilation=[dilation, dilation], padding=0, stride=stride)

    def forward(self, x, eps=1e-6):
        batch, channels, height, width = x.size()
        res = x
        x = self.head_conv(x)
        # x = F.sigmoid(x)
        if torch.max(x).cpu().detach().numpy() < 1e-16:
            print(torch.max(x).cpu().detach().numpy())
        x = F.relu(x, inplace=True)
        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding], mode='replicate')
        X = self.unfold(padded_x)
        X = X.view(batch, self.out_channels, -1, height, width)

        U = X.mean(dim=2, keepdim=True).repeat(1,1,self.ranks,1,1)
        V = X.mean(dim=1, keepdim=True).repeat(1,self.ranks,1,1,1)
        with torch.no_grad():
            # UV = torch.einsum('bcrhw,brkhw -> bckhw', U, V)[:,:,self.kernel_size//2*self.kernel_size + self.kernel_size//2,:,:]
            # print(torch.mean((x-UV)*(x-UV)).cpu().detach().numpy())
            # import cv2
            # import numpy as np
            # for i in range(x.size()[1]):
            #     cv2.imshow('pre', x.cpu().numpy()[0,i,:,:]/np.amax(x.cpu().numpy()[0,i,:,:]))
            #     cv2.imshow('post', UV.cpu().numpy()[0,i,:,:]/np.amax(UV.cpu().numpy()[0,i,:,:]))
            #     cv2.imshow('X', X.cpu().numpy()[0,i,1,:,:]/np.amax(X.cpu().numpy()[0,i,1,:,:]))
            #     cv2.waitKey()

            for _ in range(self.stage_num-1):
                numerator = torch.einsum('bcrhw,bckhw -> brkhw', U, X)
                denominator = torch.einsum('bcrhw,bckhw -> brkhw', U, torch.einsum('bcrhw,brkhw -> bckhw', U, V))
                V = V * torch.div(numerator, denominator+eps)
                # V = torch.where(V>self.eps_V, V, self.eps_V)

                numerator = torch.einsum('bckhw,brkhw -> bcrhw', X, V)
                denominator = torch.einsum('bckhw,brkhw -> bcrhw', torch.einsum('bcrhw,brkhw -> bckhw', U, V), V)
                U = U * torch.div(numerator, denominator+eps)
                # U = torch.where(U>self.eps_U, U, self.eps_U)

                UV = torch.einsum('bcrhw,brkhw -> bckhw', U, V)[:,:,self.kernel_size//2*self.kernel_size + self.kernel_size//2,:,:]
                print(torch.mean((x-UV)*(x-UV)).cpu().detach().numpy())

        numerator = torch.einsum('bcrhw,bckhw -> brkhw', U, X)
        denominator = torch.einsum('bcrhw,bckhw -> brkhw', U, torch.einsum('bcrhw,brkhw -> bckhw', U, V))
        V = V * torch.div(numerator, denominator+eps)
        # V = torch.where(V>self.eps_V, V, self.eps_V)

        numerator = torch.einsum('bckhw,brkhw -> bcrhw', X, V)
        denominator = torch.einsum('bckhw,brkhw -> bcrhw', torch.einsum('bcrhw,brkhw -> bckhw', U, V), V)
        U = U * torch.div(numerator, denominator+eps)
        # U = torch.where(U>self.eps_U, U, self.eps_U)
        # UV = torch.einsum('bcrhw,brkhw -> bckhw', U, V)[:,:,self.kernel_size//2*self.kernel_size + self.kernel_size//2,:,:]
        # print(torch.mean((x-UV)*(x-UV)).cpu().detach().numpy())
        # print('U', torch.max(U), torch.min(U))
        # print('V', torch.max(V), torch.min(V))

        center = self.kernel_size//2*self.kernel_size + self.kernel_size//2
        UV = torch.einsum('bcrhw,brkhw -> bckhw', U, V[:,:,center:center+1,:,:])
        UV = UV[:,:,0,:,:]
        # UV = UV.view(batch, self.out_channels, duration, height, width, self.kernel_size, self.kernel_size, self.duration_kernel_size).contiguous()
        # UV = UV[:,:,:,:,:,self.kernel_size//2,self.kernel_size//2,self.duration_kernel_size//2]
        # UV = UV.view(batch, self.out_channels, duration, height, width).contiguous()
        # print(torch.max(UV).cpu().detach().numpy())

        out = self.tail_conv(UV)+res
        # out = self.tail_conv(torch.cat([UV, res], dim=1))
        # out = F.relu(out, inplace=True)
        # print(torch.max(out).cpu().detach().numpy())
        # import cv2
        # import numpy as np
        # for i in range(x.size()[1]):
        #     # if i == 9:
        #     #     cv2.imwrite('pre{}.png'.format(i), x.abs().cpu().numpy()[0,i,:,:]/np.amax(x.cpu().numpy()[0,i,:,:])*255)
        #     #     cv2.imwrite('post{}.png'.format(i), UV.abs().cpu().numpy()[0,i,:,:]/np.amax(x.cpu().numpy()[0,i,:,:])*255)
        #     cv2.imshow('pre', x.abs().cpu().numpy()[0,i,:,:]/np.amax(x.cpu().numpy()[0,i,:,:]))
        #     cv2.imshow('post', UV.abs().cpu().numpy()[0,i,:,:]/np.amax(x.cpu().numpy()[0,i,:,:]))
        #     cv2.waitKey()

        return out.contiguous()

class NonLocalLayer_dilation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, bias=False, ranks=3, stage_num=1):
        super(NonLocalLayer_dilation, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.head_conv_k = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.head_conv_q = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.head_conv_v = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.tail_conv = nn.Conv2d(out_channels, in_channels, kernel_size=1, bias=bias)

        self.unfold = nn.Unfold(kernel_size=[kernel_size, kernel_size], dilation=[dilation, dilation], padding=0, stride=stride)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, eps=1e-6):
        batch, channels, height, width = x.size()
        res = x
        K = self.head_conv_k(x)
        K = F.relu(K, inplace=True)
        K = F.pad(K, [self.padding, self.padding, self.padding, self.padding], mode='replicate')
        K = self.unfold(K)
        K = K.view(batch, self.out_channels, -1, height, width)

        Q = self.head_conv_q(x)
        Q = F.relu(Q, inplace=True)
        Q = Q.view(batch, self.out_channels, 1, height, width)

        V = self.head_conv_v(x)
        V = F.relu(V, inplace=True)
        V = F.pad(V, [self.padding, self.padding, self.padding, self.padding], mode='replicate')
        V = self.unfold(V)
        V = V.view(batch, self.out_channels, -1, height, width)

        A = torch.einsum('bckhw,bcqhw -> bkqhw', K, Q)
        A = self.softmax(A)

        out = torch.einsum('bckhw,bkqhw -> bcqhw', V, A)
        out = out.view(batch, self.out_channels, height, width)
        out = self.tail_conv(out) + res
        # print(torch.max(out).cpu().detach().numpy())
        # import cv2
        # import numpy as np
        # for i in range(x.size()[1]):
        #     # if i == 9:
        #     #     cv2.imwrite('pre{}.png'.format(i), x.abs().cpu().numpy()[0,i,:,:]/np.amax(x.cpu().numpy()[0,i,:,:])*255)
        #     #     cv2.imwrite('post{}.png'.format(i), UV.abs().cpu().numpy()[0,i,:,:]/np.amax(x.cpu().numpy()[0,i,:,:])*255)
        #     cv2.imshow('pre', x.abs().cpu().numpy()[0,i,:,:]/np.amax(x.cpu().numpy()[0,i,:,:]))
        #     cv2.imshow('post', UV.abs().cpu().numpy()[0,i,:,:]/np.amax(x.cpu().numpy()[0,i,:,:]))
        #     cv2.waitKey()

        return out.contiguous()

class LowRankLayer_Lstep_dilation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, bias=False, ranks=3, stage_num=9):
        super(LowRankLayer_Lstep_dilation, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.stage_num = stage_num
        self.ranks = ranks

        self.eps_U = None
        self.eps_V = None

        self.head_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.tail_conv = nn.Conv2d(out_channels, in_channels, kernel_size=1, bias=bias)

        self.unfold = nn.Unfold(kernel_size=[kernel_size, kernel_size], dilation=[dilation, dilation], padding=0, stride=stride)

        self.alpha = [nn.Parameter(torch.tensor(1e-2), requires_grad=True) for _ in range(stage_num)]
        self.beta = [nn.Parameter(torch.tensor(1e-2), requires_grad=True) for _ in range(stage_num)]
        for i in range(stage_num):
            self.register_parameter('alpha{}'.format(i), self.alpha[i])
            self.register_parameter('beta{}'.format(i), self.beta[i])

    def forward(self, x, eps=1e-6):
        batch, channels, height, width = x.size()
        x = self.head_conv(x)
        x = F.relu(x, inplace=True)
        print(torch.max(x).cpu().detach().numpy())
        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding], mode='replicate')
        X = self.unfold(padded_x)
        X = X.view(batch, self.out_channels, height, width, -1)

        U = X.mean(dim=-1, keepdim=True).repeat(1,1,1,1,self.ranks)
        V = X.mean(dim=1, keepdim=True).repeat(1,self.ranks,1,1,1)

        for i in range(self.stage_num):
            numerator_V = torch.einsum('bchwr,bchwk -> brhwk', U, X)
            denominator_V = torch.einsum('bchwr,bchwk -> brhwk', U, torch.einsum('bchwr,brhwk -> bchwk', U, V))
            V = V - self.beta[i] * (denominator_V - numerator_V)

            numerator_U = torch.einsum('bchwk,brhwk -> bchwr', X, V)
            denominator_U = torch.einsum('bchwk,brhwk -> bchwr', torch.einsum('bchwr,brhwk -> bchwk', U, V), V)
            U = U - self.alpha[i] * (denominator_U - numerator_U)

            UV = torch.einsum('bchwr,brhwk -> bchwk', U, V)[:,:,:,:,self.kernel_size//2*self.kernel_size + self.kernel_size//2]
            print(torch.mean((x-UV)*(x-UV)).cpu().detach().numpy(), self.alpha[i].cpu().detach().numpy(), self.beta[i].cpu().detach().numpy())

        # print('U', torch.max(U), torch.min(U))
        # print('V', torch.max(V), torch.min(V))

        UV = torch.einsum('bchwr,brhwk -> bchwk', U, V)
        UV = UV[:,:,:,:,self.kernel_size//2*self.kernel_size + self.kernel_size//2]

        out = self.tail_conv(UV)#+res
        out = F.relu(out, inplace=True)
        print(torch.max(out).cpu().detach().numpy())

        return out.contiguous()

class LowRankLayer_global(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, ranks=3, stage_num=1):
        super(LowRankLayer_global, self).__init__()
        self.out_channels = out_channels
        self.stage_num = stage_num
        self.ranks = ranks

        self.eps_U = None
        self.eps_V = None

        self.head_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.tail_conv = nn.Conv2d(out_channels, in_channels, kernel_size=1, bias=bias)

    def forward(self, x, eps=1e-6):
        batch, channels, height, width = x.size()
        res = x
        x = self.head_conv(x)
        x = F.relu(x, inplace=True)
        X = x.view(batch, self.out_channels, height*width)

        U = X.mean(dim=2, keepdim=True).repeat(1,1,self.ranks)
        V = X.mean(dim=1, keepdim=True).repeat(1,self.ranks,1)
        with torch.no_grad():
            for _ in range(self.stage_num-1):
                numerator = torch.einsum('bcr,bcn -> brn', U, X)
                denominator = torch.einsum('bcr,bcn -> brn', U, torch.einsum('bcr,brn -> bcn', U, V))
                V = V * torch.div(numerator, denominator+eps)
                # V = torch.where(V>self.eps_V, V, self.eps_V)

                numerator = torch.einsum('bcn,brn -> bcr', X, V)
                denominator = torch.einsum('bcn,brn -> bcr', torch.einsum('bcr,brn -> bcn', U, V), V)
                U = U * torch.div(numerator, denominator+eps)
                # U = torch.where(U>self.eps_U, U, self.eps_U)

                # UV = torch.einsum('bcr,brn -> bcn', U, V)
                # print(torch.mean((x-UV)*(x-UV)).cpu().detach().numpy())

        numerator = torch.einsum('bcr,bcn -> brn', U, X)
        denominator = torch.einsum('bcr,bcn -> brn', U, torch.einsum('bcr,brn -> bcn', U, V))
        V = V * torch.div(numerator, denominator+eps)
        # V = torch.where(V>self.eps_V, V, self.eps_V)

        numerator = torch.einsum('bcn,brn -> bcr', X, V)
        denominator = torch.einsum('bcn,brn -> bcr', torch.einsum('bcr,brn -> bcn', U, V), V)
        U = U * torch.div(numerator, denominator+eps)
        # U = torch.where(U>self.eps_U, U, self.eps_U)
        # UV = torch.einsum('bcr,brn -> bcn', U, V)
        # print(torch.mean((x-UV)*(x-UV)).cpu().detach().numpy())
        # print('U', torch.max(U), torch.min(U))
        # print('V', torch.max(V), torch.min(V))

        UV = torch.einsum('bcr,brn -> bcn', U, V)
        UV = UV.view(batch, self.out_channels, height, width).contiguous()
        # print(torch.max(UV).cpu().detach().numpy())

        out = self.tail_conv(UV)+res
        # out = F.relu(out, inplace=True)
        # print(torch.max(out).cpu().detach().numpy())
        # import cv2
        # import numpy as np
        # for i in range(x.size()[1]):
        #     cv2.imshow('pre', x.abs().cpu().numpy()[0,i,:,:]/np.amax(x.cpu().numpy()[0,i,:,:]))
        #     cv2.imshow('post', UV.abs().cpu().numpy()[0,i,:,:]/np.amax(x.cpu().numpy()[0,i,:,:]))
        #     cv2.waitKey()

        return out

class LowRankSTLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, duration_kernel_size, stride=1, dilation=1, padding=0, duration_padding=0, bias=False, ranks=3):
        super(LowRankSTLayer, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.duration_kernel_size = duration_kernel_size
        self.stride = stride
        self.padding = padding
        self.duration_padding = duration_padding
        self.dilation = dilation

        self.head_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.tail_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)

        # self.unfold = nn.Unfold(kernel_size=[duration_kernel_size, kernel_size, kernel_size], dilation=[1, dilation, dilation], padding=0, stride=stride)

        self.U_w = nn.Parameter(torch.randn(duration_kernel_size*kernel_size**2, ranks), requires_grad=True)
        self.V_w = nn.Parameter(torch.randn(out_channels, ranks), requires_grad=True)
        
        init.normal_(self.U_w, 0, math.sqrt(2. / ranks))
        init.normal_(self.V_w, 0, math.sqrt(2. / ranks))

    def forward(self, x):
        batch, channels, duration, height, width = x.size()

        x = self.head_conv(x)
        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding, self.duration_padding, self.duration_padding], mode='replicate')

        U = padded_x.unfold(2, self.duration_kernel_size, self.stride).unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)
        # U = self.unfold(padded_x)
        
        U = U.contiguous().view(batch, self.out_channels, duration, height, width, -1)

        V = x.view(batch, self.out_channels, duration, height, width, 1)

        U = torch.einsum('bcdhwk,kr -> bcdhwr', U, self.U_w)
        # U = F.relu(U, inplace=True)
        U = self._l2norm(U, dim=-1)
        V = torch.einsum('bcdhwk,cr -> brdhwk', V, self.V_w)
        # V = F.relu(V, inplace=True)
        UV = torch.einsum('bcdhwr,brdhwk -> bcdhwk', U, V)
        UV = UV.view(batch, self.out_channels, duration, height, width)

        out = self.tail_conv(UV)

        return out

class LowRankSTLayer_dilation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, duration_kernel_size, stride=1, dilation=1, padding=0, duration_padding=0, bias=False, ranks=3, stage_num=3):
        super(LowRankSTLayer_dilation, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.duration_kernel_size = duration_kernel_size
        self.stride = stride
        self.padding = padding
        self.duration_padding = duration_padding
        self.dilation = dilation
        self.stage_num = stage_num
        self.ranks = ranks

        self.eps_U = None
        self.eps_V = None

        self.head_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.tail_conv = nn.Conv3d(out_channels, in_channels, kernel_size=1, bias=bias)

        self.unfold = nn.Unfold(kernel_size=[kernel_size, kernel_size], dilation=[dilation, dilation], padding=0, stride=stride)

        # self.U_w = nn.Parameter(torch.randn(duration_kernel_size*kernel_size**2, ranks), requires_grad=True)
        # # self.U_w2 = nn.Parameter(torch.randn(ranks, ranks), requires_grad=True)
        # self.V_w = nn.Parameter(torch.randn(out_channels, ranks), requires_grad=True)
        # # self.V_w2 = nn.Parameter(torch.randn(ranks, ranks), requires_grad=True)
        
        # init.normal_(self.U_w, 0, math.sqrt(2. / ranks))
        # # init.normal_(self.U_w2, 0, math.sqrt(2. / ranks))
        # init.normal_(self.V_w, 0, math.sqrt(2. / ranks))
        # # init.normal_(self.V_w2, 0, math.sqrt(2. / ranks))

    def forward(self, x, eps=1e-6):
        batch, channels, duration, height, width = x.size()
        res = x
        x = self.head_conv(x)
        x = F.relu(x, inplace=True)
        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding, self.duration_padding, self.duration_padding], mode='replicate')
        # X = x.view(batch, self.out_channels, duration, height, width, 1)
        X = self.unfold(padded_x.permute(0,2,1,3,4).contiguous().view(batch*(duration+2*self.duration_padding), self.out_channels, height+2*self.padding, width+2*self.padding))
        X = X.view(batch, duration+2*self.duration_padding, self.out_channels, self.kernel_size**2, height, width).permute(0,2,1,4,5,3).contiguous().unfold(2,self.duration_kernel_size,self.stride)
        X = X.contiguous().view(batch, self.out_channels, duration, height, width, -1)

        # U = self.unfold(padded_x.permute(0,2,1,3,4).contiguous().view(batch*(duration+2*self.duration_padding), self.out_channels, height+2*self.padding, width+2*self.padding))
        # U = U.view(batch, duration+2*self.duration_padding, self.out_channels, self.kernel_size**2, height, width).permute(0,2,1,4,5,3).contiguous().unfold(2,self.duration_kernel_size,self.stride)
        # U = U.contiguous().view(batch, self.out_channels, duration, height, width, -1)

        # V = x.view(batch, self.out_channels, duration, height, width, 1)

        # U = torch.einsum('bcdhwk,kr -> bcdhwr', U, self.U_w)
        # U = F.relu(U, inplace=True)
        # # U = torch.einsum('bcdhwk,kr -> bcdhwr', U, self.U_w2)
        # # U = self._l2norm(U, dim=-1)
        # V = torch.einsum('bcdhwk,cr -> brdhwk', V, self.V_w)
        # V = F.relu(V, inplace=True)
        # # V = torch.einsum('bcdhwk,cr -> brdhwk', V, self.V_w2)

        with torch.no_grad():
            U = X.mean(dim=-1, keepdim=True).repeat(1,1,1,1,1,self.ranks)
            V = X.mean(dim=1, keepdim=True).repeat(1,self.ranks,1,1,1,1)

            # if self.eps_U == None:
            #     self.eps_U = torch.ones_like(U)*eps
            # if self.eps_V == None:
            #     self.eps_V = torch.ones_like(V)*eps
            for _ in range(self.stage_num-1):
                numerator = torch.einsum('bcdhwr,bcdhwk -> brdhwk', U, X)
                denominator = torch.einsum('bcdhwr,bcdhwk -> brdhwk', U, torch.einsum('bcdhwr,brdhwk -> bcdhwk', U, V))
                V = V * torch.div(numerator, denominator+eps)
                # V = torch.where(V>self.eps_V, V, self.eps_V)

                numerator = torch.einsum('bcdhwk,brdhwk -> bcdhwr', X, V)
                denominator = torch.einsum('bcdhwk,brdhwk -> bcdhwr', torch.einsum('bcdhwr,brdhwk -> bcdhwk', U, V), V)
                U = U * torch.div(numerator, denominator+eps)
                # U = torch.where(U>self.eps_U, U, self.eps_U)

        numerator = torch.einsum('bcdhwr,bcdhwk -> brdhwk', U, X)
        denominator = torch.einsum('bcdhwr,bcdhwk -> brdhwk', U, torch.einsum('bcdhwr,brdhwk -> bcdhwk', U, V))
        V = V * torch.div(numerator, denominator+eps)
        # V = torch.where(V>self.eps_V, V, self.eps_V)

        numerator = torch.einsum('bcdhwk,brdhwk -> bcdhwr', X, V)
        denominator = torch.einsum('bcdhwk,brdhwk -> bcdhwr', torch.einsum('bcdhwr,brdhwk -> bcdhwk', U, V), V)
        U = U * torch.div(numerator, denominator+eps)
        # U = torch.where(U>self.eps_U, U, self.eps_U)
        # print('U', torch.max(U), torch.min(U))
        # print('V', torch.max(V), torch.min(V))

        UV = torch.einsum('bcdhwr,brdhwk -> bcdhwk', U, V)
        # UV = UV[:,:,:,:,:,0]
        UV = UV[:,:,:,:,:,(self.kernel_size//2*self.kernel_size + self.kernel_size//2)*self.duration_kernel_size + self.duration_kernel_size//2]
        # UV = UV.view(batch, self.out_channels, duration, height, width, self.kernel_size, self.kernel_size, self.duration_kernel_size).contiguous()
        # UV = UV[:,:,:,:,:,self.kernel_size//2,self.kernel_size//2,self.duration_kernel_size//2]
        # UV = UV.view(batch, self.out_channels, duration, height, width).contiguous()

        out = self.tail_conv(UV)
        out = F.relu(out, inplace=True)

        return out
    
    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.
        Returns a tensor where each sub-tensor of input along the given dim is 
        normalized such that the 2-norm of the sub-tensor is equal to 1.
        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the sub-tensors.
        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

class EMAU(nn.Module):
    '''The Expectation-Maximization Attention Unit (EMAU).
    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''
    def __init__(self, g, c, c_in=None, kernel_size=1, k=50, stage_num=1, momentum=0.9):
        super(EMAU, self).__init__()
        self.stage_num = stage_num
        self.momentum = momentum
        self.g = g

        if not c_in: c_in = c

        mu = torch.Tensor(1, c//g, k)
        mu.normal_(0, math.sqrt(2. / k))    # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)

        self.conv1 = nn.Conv2d(c_in, c, kernel_size, groups=g, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(c, c_in, kernel_size, groups=g, padding=kernel_size//2)       
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x, validation=False):
        idn = x
        # The first conv
        x = self.conv1(x)

        # The EM Attention
        b, c, h, w = x.size()
        x = x.view(b, self.g, c//self.g, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b, c//self.g, self.g*h*w)
        # x = x.view(b, c, h*w)               # b * c * n
        mu = self.mu.repeat(b, 1, 1)        # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)    # b * n * c
                z = torch.bmm(x_t, mu)      # b * n * k
                z = F.softmax(z, dim=2)     # b * n * k
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)       # b * c * k
                mu = self._l2norm(mu, dim=1)
            
            if not validation:
                self.mu *= self.momentum
                self.mu += mu.mean(dim=0, keepdim=True) * (1 - self.momentum)
                
        # !!! The moving averaging operation is writtern in train.py, which is significant.

        z_t = z.permute(0, 2, 1)            # b * k * n
        x = mu.matmul(z_t)                  # b * c * n
        # x = x.view(b, c, h, w)              # b * c * h * w
        x = x.view(b, c//self.g, self.g, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b, c, h, w)
        x = F.relu(x, inplace=True)

        # The second conv
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)

        return x

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.
        Returns a tensor where each sub-tensor of input along the given dim is 
        normalized such that the 2-norm of the sub-tensor is equal to 1.
        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the sub-tensors.
        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))
        

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

def pixel_unshuffle(input, upscale_factor):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.
    written by: Zhaoyi Yan, https://github.com/Zhaoyi-Yan
    and Kai Zhang, https://github.com/cszn/FFDNet
    01/01/2019
    """
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


class PixelUnShuffle(nn.Module):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.
    written by: Zhaoyi Yan, https://github.com/Zhaoyi-Yan
    and Kai Zhang, https://github.com/cszn/FFDNet
    01/01/2019
    """

    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)


if __name__ == '__main__':
    # x = torch.cuda.FloatTensor(16,8,256,256).fill_(1)
    # net = LowRankLayer(8, 16, 7, padding=7//2).cuda()
    x = torch.FloatTensor(10,16,8,128,128).fill_(1)
    net = LowRankSTLayer(16, 8, 5, 3, padding=5//2, duration_padding=3//2)
    net = LowRankSTLayer_dilation(16, 8, 5, 3, dilation=2, padding=(2*5-2)//2, duration_padding=3//2)
    y = net(x)
    import ipdb; ipdb.set_trace()