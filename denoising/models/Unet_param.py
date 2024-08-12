import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

if __name__ == "__main__":
    import common
else:
    from models import common


class ParalConv(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, mask_in_ch=1, mask_out_ch=1, kernel_size=5, stride=1, n_feat=6):
        super(ParalConv, self).__init__()

        self.mask_transform = nn.Conv2d(mask_in_ch, mask_out_ch, kernel_size=2*kernel_size-1, stride=stride, padding=(2*kernel_size-1-stride)//2, bias=True)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=(kernel_size-stride)//2, bias=True)
        # self.mask_norm = nn.InstanceNorm2d(mask_out_ch)

    def forward(self, x, mask):
        # mask = self.mask_norm(mask)
        x = self.conv(x)
        mask = self.mask_transform(mask)
        # mask = self.mask_norm(mask)

        return x, mask


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
        ref_normed = self.norm_layer(ref)
        if self.learnable:
            style = self.conv_shared(torch.cat([lr, ref], dim=1))
            gamma = self.conv_gamma(style)
            beta = self.conv_beta(style)

        b, c, h, w = lr.size()
        lr = lr.view(b, c, h * w)
        lr_mean = torch.mean(lr, dim=-1, keepdim=True).unsqueeze(3)
        lr_std = torch.std(lr, dim=-1, keepdim=True).unsqueeze(3)

        if self.learnable:
            if self.use_residual:
                gamma = gamma + lr_std
                beta = beta + lr_mean
            else:
                gamma = 1 + gamma
        else:
            gamma = lr_std
            beta = lr_mean

        out = ref_normed * gamma + beta

        return out


class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch, mask_in_ch=None, mask_out_ch=None, use_sam=False):
        super(conv_block, self).__init__()

        if mask_in_ch is None:
            mask_in_ch = in_ch
        if mask_out_ch is None:
            mask_out_ch = out_ch

        # self.conv1 = ParalConv(in_ch, out_ch, mask_in_ch, mask_out_ch, kernel_size=3, stride=1)
        self.conv1 = ParalConv(in_ch, out_ch, mask_in_ch, mask_out_ch, kernel_size=3, stride=1)
        self.conv2 = ParalConv(out_ch, out_ch, mask_out_ch, mask_out_ch, kernel_size=3, stride=1)
        self.fuse = nn.Conv2d(out_ch+mask_out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        # self.sam = SAM(out_ch)
        
        self.conv_residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True)

        # self.use_sam = use_sam
        # if self.use_sam:
        # self.sam = SAM(out_ch, mask_out_ch)

        self.out_ch = out_ch
        # self.norm = nn.LayerNorm([1,2,3])
        # self.norm = nn.InstanceNorm2d(mask_out_ch)

    def forward(self, x, mask):
        res = self.conv_residual(x)

        x, mask = self.conv1(x, mask)
        x, mask = self.act(x), self.act(mask)
        x, mask = self.conv2(x, mask)
        x, mask = self.act(x), self.act(mask)
        # x = self.act(x)
        # x = mask[:, :self.out_ch, :, :] * x + mask[:, self.out_ch:, :, :]
        # mask = self.act(self.fuse(mask))
        # x = res + x

        # if self.use_sam:
        # m = self.sam(x, mask)
        # x = torch.cat([x, m], dim=1)
        # else:
        # mask = self.norm(mask)
        # x = torch.cat([x, mask], dim=1)
        # x = self.fuse(x)

        return x+res, mask
        # mask = self.mask_norm(mask)

        # return x+res, mask


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
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, mask_in_ch=None, mask_out_ch=None, kernel_size=4, stride=2):
        super(Down, self).__init__()
        if mask_in_ch is None:
            mask_in_ch = in_ch
        if mask_out_ch is None:
            mask_out_ch = out_ch
        self.mask_transform = nn.Conv2d(mask_in_ch, mask_out_ch, kernel_size=kernel_size, stride=stride, padding=(kernel_size-stride)//2, bias=True)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=(kernel_size-stride)//2, bias=True)
    
    def forward(self, x, mask):
        x = self.conv(x)
        mask = self.mask_transform(mask)
        return x, mask

class Up(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, mask_in_ch=None, mask_out_ch=None, kernel_size=3, stride=1):
        super(Up, self).__init__()
        if mask_in_ch is None:
            mask_in_ch = in_ch
        if mask_out_ch is None:
            mask_out_ch = out_ch
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.mask_transform = nn.Conv2d(mask_in_ch, mask_out_ch, kernel_size=kernel_size, stride=stride, padding=(kernel_size-stride)//2, bias=True)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=(kernel_size-stride)//2, bias=True)
    
    def forward(self, x, mask):
        x = self.conv(self.up(x))
        mask = self.mask_transform(self.up(mask))
        return x, mask



class U_Net_param(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=34, out_ch=34, mask_in_ch=25, epoch_thres=20):
        super(U_Net_param, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        m1 = 64
        m_filters = [m1, m1 * 2, m1 * 4, m1 * 8, m1 * 16]
        
        self.Down1 = Down(filters[0], filters[0], m_filters[0], m_filters[0])
        self.Down2 = Down(filters[1], filters[1], m_filters[1], m_filters[1])
        self.Down3 = Down(filters[2], filters[2], m_filters[2], m_filters[2])
        self.Down4 = Down(filters[3], filters[3], m_filters[3], m_filters[3])

        self.Conv1 = conv_block(in_ch, filters[0], mask_in_ch, m_filters[0])
        self.Conv2 = conv_block(filters[0], filters[1], m_filters[0], m_filters[1])
        self.Conv3 = conv_block(filters[1], filters[2], m_filters[1], m_filters[2])
        self.Conv4 = conv_block(filters[2], filters[3], m_filters[2], m_filters[3])
        self.Conv5 = conv_block(filters[3], filters[4], m_filters[3], m_filters[4])

        self.Up5 = Up(filters[4], filters[3], m_filters[4], m_filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3], m_filters[4], m_filters[3])

        self.Up4 = Up(filters[3], filters[2], m_filters[3], m_filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2], m_filters[3], m_filters[2])

        self.Up3 = Up(filters[2], filters[1], m_filters[2], m_filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1], m_filters[2], m_filters[1])

        self.Up2 = Up(filters[1], filters[0], m_filters[1], m_filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0], m_filters[1], m_filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.mask_in_ch = mask_in_ch

        self.epoch_thres = epoch_thres


    def forward(self, x, mask, epoch):

        # if epoch < self.epoch_thres:
        #     mask = torch.rand_like(mask)
        mask = torch.zeros_like(mask)

        # import pdb; pdb.set_trace()
        # e1, me1 = self.Conv1(torch.cat([x, mask], dim=1), mask)
        e1, me1 = self.Conv1(x, mask)

        e2, me2 = self.Down1(e1, me1)
        e2, me2 = self.Conv2(e2, me2)

        e3, me3 = self.Down2(e2, me2)
        e3, me3 = self.Conv3(e3, me3)

        e4, me4 = self.Down3(e3, me3)
        e4, me4 = self.Conv4(e4, me4)

        e5, me5 = self.Down4(e4, me4)
        e5, me5 = self.Conv5(e5, me5)

        d5, md5 = self.Up5(e5, me5)
        d5, md5 = torch.cat((e4, d5), dim=1), torch.cat((me4, md5), dim=1)
        d5, md5 = self.Up_conv5(d5, md5)

        d4, md4 = self.Up4(d5, md5)
        d4, md4 = torch.cat((e3, d4), dim=1), torch.cat((me3, md4), dim=1)
        d4, md4 = self.Up_conv4(d4, md4)

        d3, md3 = self.Up3(d4, md4)
        d3, md3 = torch.cat((e2, d3), dim=1), torch.cat((me2, md3), dim=1)
        d3, md3 = self.Up_conv3(d3, md3)

        d2, md2 = self.Up2(d3, md3)
        d2, md2 = torch.cat((e1, d2), dim=1), torch.cat((me1, md2), dim=1)
        d2, md2 = self.Up_conv2(d2, md2)

        out = self.Conv(d2)


        return out+x


