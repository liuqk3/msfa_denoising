import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

if __name__ == "__main__":
    import common
else:
    from models import common


class Modulate(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()

        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, cond):

        # Part 2. produce scaling and bias conditioned on semantic map
        cond = F.interpolate(cond, size=x.size()[2:], mode='bilinear')
        actv = self.mlp_shared(cond)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = x * (1 + gamma) + beta

        return out


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




class U_Net_Mod(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=34, out_ch=34):
        super(U_Net_Mod, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Down1 = nn.Conv2d(filters[0], filters[0], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down2 = nn.Conv2d(filters[1], filters[1], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down3 = nn.Conv2d(filters[2], filters[2], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down4 = nn.Conv2d(filters[3], filters[3], kernel_size=4, stride=2, padding=1, bias=True)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.mod1 = Modulate(filters[0], filters[0])

        self.Conv2 = conv_block(filters[0], filters[1])
        self.mod2 = Modulate(filters[1], filters[1])

        self.Conv3 = conv_block(filters[1], filters[2])
        self.mod3 = Modulate(filters[2], filters[2])

        self.Conv4 = conv_block(filters[2], filters[3])
        self.mod4 = Modulate(filters[3], filters[3])

        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv5 = conv_block(filters[4], filters[3])
        self.up_mod5 = Modulate(filters[3], filters[3])

        self.Up4 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv4 = conv_block(filters[3], filters[2])
        self.up_mod4 = Modulate(filters[2], filters[2])

        self.Up3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv3 = conv_block(filters[2], filters[1])
        self.up_mod3 = Modulate(filters[1], filters[1])

        self.Up2 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv2 = conv_block(filters[1], filters[0])
        self.up_mod2 = Modulate(filters[0], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        e1 = self.Conv1(x)
        e1 = self.mod1(e1, x)

        e2 = self.Down1(e1)
        e2 = self.Conv2(e2)
        e2 = self.mod2(e2, x)

        e3 = self.Down2(e2)
        e3 = self.Conv3(e3)
        e3 = self.mod3(e3, x)

        e4 = self.Down3(e3)
        e4 = self.Conv4(e4)
        e4 = self.mod4(e4, x)

        e5 = self.Down4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        d5 = self.up_mod5(d5, x)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d4 = self.up_mod4(d4, x)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d3 = self.up_mod3(d3, x)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d2 = self.up_mod2(d2, x)

        out = self.Conv(d2)

        #d1 = self.active(out)

        return out+x

