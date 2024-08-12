import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import copy
from .Unet_adaptive import Ada_U_Net
from .Unet_pac import U_Net_pac
from .Unet_param import U_Net_param
from .Unet_param import SAM

if __name__ == "__main__":
    import common
else:
    from models import common


class SAM(nn.Module):
    def __init__(self, fe_ch, mask_ch, use_residual=True, learnable=True):
        super(SAM, self).__init__()

        self.learnable = learnable
        self.norm_layer = nn.InstanceNorm2d(mask_ch, affine=False)

        if self.learnable:
            self.conv_shared = nn.Sequential(nn.Conv2d(fe_ch + mask_ch, mask_ch, 3, 1, 1, bias=True),
                                             nn.ReLU(inplace=True))
            self.conv_gamma = nn.Conv2d(mask_ch, mask_ch, 3, 1, 1, bias=True)
            self.conv_beta = nn.Conv2d(mask_ch, mask_ch, 3, 1, 1, bias=True)

            self.use_residual = use_residual

            # initialization
            self.conv_gamma.weight.data.zero_()
            self.conv_beta.weight.data.zero_()
            self.conv_gamma.bias.data.zero_()
            self.conv_beta.bias.data.zero_()

    def forward(self, lr, ref):
        # ref_normed = self.norm_layer(ref)
        b, c, h, w = lr.size()
        ref_view = ref.view(b, -1)
        ref_mean = torch.mean(ref_view, dim=-1, keepdim=True)
        ref_std = torch.std(ref_view, dim=-1, keepdim=True)
        ref_normed = (ref_view - ref_mean) / ref_std

        if self.learnable:
            style = self.conv_shared(torch.cat([lr, ref], dim=1))
            gamma = self.conv_gamma(style)
            beta = self.conv_beta(style)

        b, c, h, w = lr.size()
        lr = lr.view(b, c * h * w)
        lr_mean = torch.mean(lr, dim=-1, keepdim=True)
        lr_std = torch.std(lr, dim=-1, keepdim=True)

        if self.learnable:
            gamma = gamma + lr_std.unsqueeze(-1).unsqueeze(-1)
            beta = beta + lr_mean.unsqueeze(-1).unsqueeze(-1)

        else:
            gamma = lr_std.unsqueeze(-1).unsqueeze(-1)
            beta = lr_mean.unsqueeze(-1).unsqueeze(-1)

        out = ref_normed.view(b, ref.shape[1], h, w) * gamma + beta

        return out


class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch, norm=False):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True))
        
        self.conv_residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True)

        self.norm = norm
        if norm == True:
            self.insnorm = nn.InstanceNorm2d(out_ch, affine=False)            

    def forward(self, x):

        x = self.conv(x) + self.conv_residual(x)
        if self.norm:
            x = self.insnorm(x)
        return x




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

class U_Net_cb(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=34, out_ch=34):
        super(U_Net_cb, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Down1 = nn.Conv2d(filters[0], filters[0], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down2 = nn.Conv2d(filters[1], filters[1], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down3 = nn.Conv2d(filters[2], filters[2], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down4 = nn.Conv2d(filters[3], filters[3], kernel_size=4, stride=2, padding=1, bias=True)

        self.Conv1 = conv_block(in_ch*4, filters[0])
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

    def forward(self, x, cb_param):

        e1 = self.Conv1(torch.cat([x, cb_param], dim=1))

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


class U_Net_modulate(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=34, out_ch=34):
        super(U_Net_modulate, self).__init__()

        # n1 = 64
        n1 = 1
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        # self.fuse = nn.Conv2d(in_ch*3, in_ch, kernel_size=1, stride=1, padding=0, bias=True)
        # self.conv_a = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, groups=in_ch, bias=True)
        # self.conv_b = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, groups=in_ch, bias=True)
        # self.conv_c = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, groups=in_ch, bias=True)

        # self.mask_norm = nn.InstanceNorm2d(in_ch)

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

        self.in_ch = in_ch

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

        # out = self.Conv(d2)

        #d1 = self.active(out)

        return e1, e2, e3, e4, e5, d5, d4, d3, d2


class U_Net_cbv2(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=34, out_ch=34, mask_in_ch=25):
        super(U_Net_cbv2, self).__init__()

        self.mod_net = U_Net_modulate(mask_in_ch, 1)

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

        self.e1_fuse = nn.Conv2d(1+filters[0], filters[0], kernel_size=1, stride=1, padding=0, bias=True)
        self.e2_fuse = nn.Conv2d(2+filters[1], filters[1], kernel_size=1, stride=1, padding=0, bias=True)
        self.e3_fuse = nn.Conv2d(4+filters[2], filters[2], kernel_size=1, stride=1, padding=0, bias=True)
        self.e4_fuse = nn.Conv2d(8+filters[3], filters[3], kernel_size=1, stride=1, padding=0, bias=True)
        self.e5_fuse = nn.Conv2d(16+filters[4], filters[4], kernel_size=1, stride=1, padding=0, bias=True)
        self.d5_fuse = nn.Conv2d(8+filters[3], filters[3], kernel_size=1, stride=1, padding=0, bias=True)
        self.d4_fuse = nn.Conv2d(4+filters[2], filters[2], kernel_size=1, stride=1, padding=0, bias=True)
        self.d3_fuse = nn.Conv2d(2+filters[1], filters[1], kernel_size=1, stride=1, padding=0, bias=True)
        self.d2_fuse = nn.Conv2d(1+filters[0], filters[0], kernel_size=1, stride=1, padding=0, bias=True)

        # self.e1_sam = SAM(filters[0], filters[0], learnable=False)
        # self.e2_sam = SAM(filters[1], filters[1], learnable=False)
        # self.e3_sam = SAM(filters[2], filters[2], learnable=False)
        # self.e4_sam = SAM(filters[3], filters[3], learnable=False)
        # self.e5_sam = SAM(filters[4], filters[4], learnable=False)
        # self.d5_sam = SAM(filters[3], filters[3], learnable=False)
        # self.d4_sam = SAM(filters[2], filters[2], learnable=False)
        # self.d3_sam = SAM(filters[1], filters[1], learnable=False)
        # self.d2_sam = SAM(filters[0], filters[0], learnable=False)


    def forward(self, x, cb_param):
        e1_cb, e2_cb, e3_cb, e4_cb, e5_cb, d5_cb, d4_cb, d3_cb, d2_cb = self.mod_net(cb_param)

        e1 = self.Conv1(x)
        # e1_cb = self.e1_sam(e1, e1_cb)
        # import pdb; pdb.set_trace()
        e1 = self.e1_fuse(torch.cat([e1, e1_cb], dim=1))

        e2 = self.Down1(e1)
        e2 = self.Conv2(e2)
        # e2_cb = self.e2_sam(e2, e2_cb)
        e2 = self.e2_fuse(torch.cat([e2, e2_cb], dim=1))

        e3 = self.Down2(e2)
        e3 = self.Conv3(e3)
        # e3_cb = self.e3_sam(e3, e3_cb)
        e3 = self.e3_fuse(torch.cat([e3, e3_cb], dim=1))

        e4 = self.Down3(e3)
        e4 = self.Conv4(e4)
        # e4_cb = self.e4_sam(e4, e4_cb)
        e4 = self.e4_fuse(torch.cat([e4, e4_cb], dim=1))

        e5 = self.Down4(e4)
        e5 = self.Conv5(e5)
        # e5_cb = self.e5_sam(e5, e5_cb)
        e5 = self.e5_fuse(torch.cat([e5, e5_cb], dim=1))

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        # d5_cb = self.d5_sam(d5, d5_cb)
        d5 = self.d5_fuse(torch.cat([d5, d5_cb], dim=1))

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        # d4_cb = self.d4_sam(d4, d4_cb)
        d4 = self.d4_fuse(torch.cat([d4, d4_cb], dim=1))

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        # d3_cb = self.d3_sam(d3, d3_cb)
        d3 = self.d3_fuse(torch.cat([d3, d3_cb], dim=1))

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        # d2_cb = self.d2_sam(d2, d2_cb)
        d2 = self.d2_fuse(torch.cat([d2, d2_cb], dim=1))

        out = self.Conv(d2)

        #d1 = self.active(out)

        return out+x


class U_Net_cbv2_norm(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=34, out_ch=34, mask_in_ch=25, norm_learnable=True):
        super(U_Net_cbv2_norm, self).__init__()

        self.mod_net = U_Net_modulate(mask_in_ch, 1)

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        m1 = 1
        m_filters = [m1, m1 * 2, m1 * 4, m1 * 8, m1 * 16]
        
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

        self.e1_fuse = nn.Conv2d(1+filters[0], filters[0], kernel_size=1, stride=1, padding=0, bias=True)
        self.e2_fuse = nn.Conv2d(2+filters[1], filters[1], kernel_size=1, stride=1, padding=0, bias=True)
        self.e3_fuse = nn.Conv2d(4+filters[2], filters[2], kernel_size=1, stride=1, padding=0, bias=True)
        self.e4_fuse = nn.Conv2d(8+filters[3], filters[3], kernel_size=1, stride=1, padding=0, bias=True)
        self.e5_fuse = nn.Conv2d(16+filters[4], filters[4], kernel_size=1, stride=1, padding=0, bias=True)
        self.d5_fuse = nn.Conv2d(8+filters[3], filters[3], kernel_size=1, stride=1, padding=0, bias=True)
        self.d4_fuse = nn.Conv2d(4+filters[2], filters[2], kernel_size=1, stride=1, padding=0, bias=True)
        self.d3_fuse = nn.Conv2d(2+filters[1], filters[1], kernel_size=1, stride=1, padding=0, bias=True)
        self.d2_fuse = nn.Conv2d(1+filters[0], filters[0], kernel_size=1, stride=1, padding=0, bias=True)

        self.e1_sam = SAM(filters[0], m_filters[0], learnable=norm_learnable)
        self.e2_sam = SAM(filters[1], m_filters[1], learnable=norm_learnable)
        self.e3_sam = SAM(filters[2], m_filters[2], learnable=norm_learnable)
        self.e4_sam = SAM(filters[3], m_filters[3], learnable=norm_learnable)
        self.e5_sam = SAM(filters[4], m_filters[4], learnable=norm_learnable)
        self.d5_sam = SAM(filters[3], m_filters[3], learnable=norm_learnable)
        self.d4_sam = SAM(filters[2], m_filters[2], learnable=norm_learnable)
        self.d3_sam = SAM(filters[1], m_filters[1], learnable=norm_learnable)
        self.d2_sam = SAM(filters[0], m_filters[0], learnable=norm_learnable)


    def forward(self, x, cb_param):
        e1_cb, e2_cb, e3_cb, e4_cb, e5_cb, d5_cb, d4_cb, d3_cb, d2_cb = self.mod_net(cb_param)

        e1 = self.Conv1(x)
        # import pdb; pdb.set_trace()
        e1_cb = self.e1_sam(e1, e1_cb)
        # import pdb; pdb.set_trace()
        e1 = self.e1_fuse(torch.cat([e1, e1_cb], dim=1))

        e2 = self.Down1(e1)
        e2 = self.Conv2(e2)
        e2_cb = self.e2_sam(e2, e2_cb)
        e2 = self.e2_fuse(torch.cat([e2, e2_cb], dim=1))

        e3 = self.Down2(e2)
        e3 = self.Conv3(e3)
        e3_cb = self.e3_sam(e3, e3_cb)
        e3 = self.e3_fuse(torch.cat([e3, e3_cb], dim=1))

        e4 = self.Down3(e3)
        e4 = self.Conv4(e4)
        e4_cb = self.e4_sam(e4, e4_cb)
        e4 = self.e4_fuse(torch.cat([e4, e4_cb], dim=1))

        e5 = self.Down4(e4)
        e5 = self.Conv5(e5)
        e5_cb = self.e5_sam(e5, e5_cb)
        e5 = self.e5_fuse(torch.cat([e5, e5_cb], dim=1))

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        d5_cb = self.d5_sam(d5, d5_cb)
        d5 = self.d5_fuse(torch.cat([d5, d5_cb], dim=1))

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d4_cb = self.d4_sam(d4, d4_cb)
        d4 = self.d4_fuse(torch.cat([d4, d4_cb], dim=1))

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d3_cb = self.d3_sam(d3, d3_cb)
        d3 = self.d3_fuse(torch.cat([d3, d3_cb], dim=1))

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d2_cb = self.d2_sam(d2, d2_cb)
        d2 = self.d2_fuse(torch.cat([d2, d2_cb], dim=1))

        out = self.Conv(d2)

        #d1 = self.active(out)

        return out+x


class UNet_double(nn.Module):
    def __init__(self, in_ch=25, out_ch=25):
        super(UNet_double, self).__init__()
        self.net1 = U_Net(25, 25)
        self.net2 = U_Net(25, 25)

    def forward(self, x):
        out1 = self.net1(x)
        # input_net2 = copy.deepcopy(out1)
        input_net2 = out1.clone().detach()
        # input_net2 = input_net2.detach()

        out2 = self.net2(input_net2)

        return out1, out2



class UNet_double_finetune(nn.Module):
    def __init__(self, in_ch=25, out_ch=25):
        super(UNet_double_finetune, self).__init__()
        self.net1 = U_Net(25, 25)
        self.net2 = U_Net(25, 25)

    def forward(self, x, x_cb=None):
        out1 = self.net1(x)
        # input_net2 = copy.deepcopy(out1)
        # input_net2 = out1.clone().detach()
        # input_net2 = input_net2.detach()

        out2 = self.net2(out1)
        # if x_cb is not None:
        #     out2 = self.net2(x_cb)
        # else:
        #     out2 = self.net2(out1)

        return out1, out2

class UNet_double_cb_finetune(nn.Module):
    def __init__(self, in_ch=25, out_ch=25):
        super(UNet_double_cb_finetune, self).__init__()
        self.net1 = U_Net(25, 25)
        self.net2 = U_Net_cb(25, 25)

    def forward(self, x, cb_param):
        out1 = self.net1(x)
        # input_net2 = copy.deepcopy(out1)
        # input_net2 = out1.clone().detach()
        # input_net2 = input_net2.detach()

        # out2 = self.net2(input_net2)
        out2 = self.net2(out1, cb_param)

        return out1, out2


class UNet_double_cbv3_finetune(nn.Module):
    def __init__(self, in_ch=25, out_ch=25):
        super(UNet_double_cbv3_finetune, self).__init__()
        self.net1 = U_Net(25, 25)
        self.net2 = Ada_U_Net(25, 25, 75, False)

    def forward(self, x, cb_param):
        out1 = self.net1(x)
        # input_net2 = copy.deepcopy(out1)
        # input_net2 = out1.clone().detach()
        # input_net2 = input_net2.detach()

        # out2 = self.net2(input_net2)
        out2 = self.net2(out1, cb_param)

        return out1, out2


class UNet_double_cbv2_finetune(nn.Module):
    def __init__(self, in_ch=25, out_ch=25):
        super(UNet_double_cbv2_finetune, self).__init__()
        self.net1 = U_Net(25, 25)
        self.net2 = U_Net_cbv2_norm(25, 25, 1)
        # self.net2 = U_Net_cbv2(25, 25, 1)

    def forward(self, x, cb_param):
        out1 = self.net1(x)
        # input_net2 = copy.deepcopy(out1)
        # input_net2 = out1.clone().detach()
        # input_net2 = input_net2.detach()

        # out2 = self.net2(input_net2)
        out2 = self.net2(out1, cb_param)

        return out1, out2

class UNet_double_cbv2_joint(nn.Module):
    def __init__(self, in_ch=25, out_ch=25):
        super(UNet_double_cbv2_joint, self).__init__()
        self.net1 = U_Net(25, 25)
        self.net2 = U_Net_cbv2_norm(25, 25, 1)
        # self.net2 = U_Net_cbv2(25, 25, 1)

    def forward(self, x, cb_param):
        out1 = self.net1(x)
        # input_net2 = copy.deepcopy(out1)
        input_net2 = out1.clone().detach()
        # input_net2 = input_net2.detach()

        # out2 = self.net2(input_net2)
        out2 = self.net2(input_net2, cb_param)

        return out1, out2


class U_Net_pos_emb(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=34, out_ch=34, pos_ch=1, H=256, W=256):
        super(U_Net_pos_emb, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pos_emb = nn.Parameter(torch.zeros(1, pos_ch, H, W))
        
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

    def forward(self, x, h, w):

        pos = self.pos_emb[:, :, h:h+x.shape[-2], w:w+x.shape[-1]]
        x = x + pos

        # e1 = self.Conv1(torch.cat([x, pos], dim=1))
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


class NoiseDecoupledNet(nn.Module):
    def __init__(self, net1, net2):
        super(NoiseDecoupledNet, self).__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, x):
        out1 = self.net1(x)
        out2 = self.net2(out1)

        return out1, out2

class NoiseDecoupledNet_PosEmb(nn.Module):
    def __init__(self, net1, net2):
        super(NoiseDecoupledNet_PosEmb, self).__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, x, h, w):
        out1 = self.net1(x)
        out2 = self.net2(out1, h, w)

        return out1, out2