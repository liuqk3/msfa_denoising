from audioop import bias
from email.errors import MisplacedEnvelopeHeaderDefect
from re import S
from turtle import forward
from matplotlib.pyplot import sca
from numpy import pad
import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_relu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, padding_mode='zeros'):
        super(conv_relu, self).__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, padding_mode=padding_mode)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class spatialBlock(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, x):
        pass


class GCN(nn.Module):
    def __init__(self, in_channel, mid_channel):
        super(GCN).__init__()

        self.conv0 = nn.Conv2d(in_channel, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=1)

    def forward(self, Vnode):
        net = self.conv0(Vnode)
        net = Vnode + net
        net = self.relu(net)
        net = torch.permute(net, dim=())

        net = self.conv1(net)
        
        return net


class GloRe(nn.Module):
    def __init__(self, in_channels):
        super(GloRe).__init__()

        N = in_channels // 4
        C = in_channels // 2

        self.conv0 = nn.Conv2d(in_channels, N, kernel_size=1, stride=1)

        self.conv1 = nn.Conv2d(in_channels, C, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=1)

        self.gcn = GCN()

        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)


    def forward(self, x):
        B = self.conv0(x)
        B = B.reshape


class BlockE(nn.Module):
    def __init__(self, channels=64):
        super(BlockE).__init__()
        self.conv1 = conv_relu(channels, channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='replicate')
        self.conv2 = conv_relu(channels, channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='replicate')
        self.conv3 = conv_relu(channels, channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='replicate')
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)

        self.G = GloRe()

    def forward(self, input):
        input_tensor = input
        conv1 = self.conv1(input_tensor)

        tmp = input_tensor, conv1
        conv2 = self.conv2(tmp)

        tmp = tmp + conv2
        conv3 = self.conv3(tmp)

        tmp = tmp + conv3
        fuse = self.conv4(tmp)

        fuse = fuse + self.G(fuse)

        return fuse + input



class BlockD(nn.Module):
    def __init__(self, channels=64):
        super(BlockD, self).__init__()
        self.spatial = spatialBlock()
        self.conv0 = conv_relu(channels, channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='replicate')
        self.conv1 = conv_relu(channels, channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='replicate')
        self.conv2 = conv_relu(channels, channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='replicate')
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, input):
        input_tensor = input + self.spatial(input)
        conv1 = self.conv0(input_tensor)

        tmp = input_tensor + conv1
        conv2 = self.conv1(tmp)

        tmp = tmp + conv2
        conv3 = self.conv2(tmp)

        tmp = input_tensor + conv3
        fuse = self.conv3(tmp)

        return fuse + input


class GRNet(nn.Module):
    def __init__(self, in_channels=25, channels=64):
        super(GRNet, self).__init__()

        self.conv0 = nn.Conv2d(25, channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='replicate')
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='replicate')

        self.encoder0 = BlockE()
        self.down0 = conv_relu(channels, channels, kernel_size=3, padding=1, stride=2, bias=True, padding_mode='replicate')

        self.encoder1 = BlockE()
        self.down1 = conv_relu(channels, channels, kernel_size=3, padding=1, stride=2, bias=True, padding_mode='replicate')

        self.encoder2 = BlockE()
        self.down2 = conv_relu(channels, channels, kernel_size=3, padding=1, stride=2, bias=True, padding_mode='replicate')

        self.middle = BlockE()

        self.conv2 = conv_relu(channels, channels, kernel_size=1, padding=1, stride=1, bias=True, padding_mode='replicate')
        self.decoder0 = BlockD()

        self.conv3 = conv_relu(channels, channels, kernel_size=1, padding=1, stride=1, bias=True, padding_mode='replicate')
        self.decoder1 = BlockD()

        self.conv4 = conv_relu(channels, channels, kernel_size=1, padding=1, stride=1, bias=True, padding_mode='replicate')
        self.decoder2 = BlockD()
        # self.conv2 = conv_relu(channels*2, channels, kernel_size=1, padding=1, stride=1, bias=True padding_mode='replicate')

        self.conv5 = nn.Conv2d(channels, in_channels, kernel_size=3, padding=1, stride=1, padding_mode='replicate')


    def forward(self, x):
        basic = self.conv0(x)
        basic1 = self.conv1(basic)

        encode0 = self.encoder0(basic1)
        down0 = self.down0(encode0)

        encode1 = self.encoder1(down0)
        down1 = self.down1(encode1)

        encode2 = self.encoder2(down1)
        down2 = self.down2(encode2)

        media_end = self.middle(down2)

        deblock0 = F.upsample_bilinear(media_end, scale_factor=2)
        deblock0 = torch.cat((deblock0, encode2), dim=1)
        deblock0 = self.conv2(deblock0)
        deblock0 = self.decoder0(deblock0)

        deblock1 = F.upsample_bilinear(deblock0, scale_factor=2)
        deblock1 = torch.cat((deblock1, encode1), dim=1)
        deblock1 = self.conv3(deblock1)
        deblock1 = self.decoder1(deblock1)

        deblock2 = F.upsample_bilinear(deblock1, scale_factor=2)
        deblock2 = torch.cat((deblock2, encode0), dim=1)
        deblock2 = self.conv4(deblock2)
        deblock2 = self.decoder2(deblock2)

        decoding_end = deblock2 + basic
        res = self.conv5(decoding_end)
        out = x + res

        return out