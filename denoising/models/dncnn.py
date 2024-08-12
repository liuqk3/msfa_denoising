# pytorch reimplementation of DnCNN (https://github.com/cszn/DnCNN)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

if __name__ == "__main__":
    import common
else:
    from models import common


def conv3x3_bn_relu(in_planes, out_planes, stride=1, inplace=True):
    """3x3 Convolution with padding - Batch Normalization - ReLu"""
    conv = nn.Conv2d(
        in_planes, out_planes, 
        kernel_size=3, stride=stride, 
        padding=1, bias=False)
    bn = nn.BatchNorm2d(num_features=out_planes)
    relu = nn.ReLU(inplace=inplace)
    return nn.Sequential(conv, bn, relu)


class DnCNN(nn.Module):
    """Args:
    bn_channels (int): number of channels used in conv_bn_relu layers
    depth (int): number of conv_bn_relu layers
    """
    def __init__(self, in_channels=1, bn_channels=64, depth=18):
        super(DnCNN, self).__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(
                in_channels, bn_channels,
                kernel_size=3, padding=1,
            ),
            nn.ReLU(inplace=True)
        )

        self.conv_bn_relu_list = nn.ModuleList(
            [conv3x3_bn_relu(bn_channels, bn_channels) for _ in range(depth)]
        )
        
        self.conv = nn.Conv2d(
            bn_channels, in_channels,
            kernel_size=3, padding=1,
        )

    def forward(self, X):
        residual = X
        out = self.conv_relu(X)
        for layer in self.conv_bn_relu_list:
            out = layer(out)
        out = self.conv(out)
        out = out + residual
        return out

class DnCNN_LR(nn.Module):
    """Args:
    bn_channels (int): number of channels used in conv_bn_relu layers
    depth (int): number of conv_bn_relu layers
    """
    def __init__(self, in_channels=1, bn_channels=64, depth=18):
        super(DnCNN_LR, self).__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(
                in_channels, bn_channels,
                kernel_size=3, padding=1,
            ),
            nn.ReLU(inplace=True)
        )

        self.conv_bn_relu_list1 = nn.ModuleList(
            [conv3x3_bn_relu(bn_channels, bn_channels) for _ in range(depth//2)]
        )

        self.lr = common.LowRankLayer_dilation(bn_channels, bn_channels//2, 5, dilation=3, padding=6, ranks=bn_channels//16)
        # self.lr = common.LowRankLayer_global(bn_channels, bn_channels, ranks=bn_channels//2, stage_num=3)

        self.conv_bn_relu_list2 = nn.ModuleList(
            [conv3x3_bn_relu(bn_channels, bn_channels) for _ in range(depth//2)]
        )
        
        self.conv = nn.Conv2d(
            bn_channels, in_channels,
            kernel_size=3, padding=1,
        )

    def forward(self, X):
        residual = X
        out = self.conv_relu(X)
        for layer in self.conv_bn_relu_list1:
            out = layer(out)
        out = self.lr(out)
        for layer in self.conv_bn_relu_list2:
            out = layer(out)
        out = self.conv(out)
        out = out + residual
        return out

class DnCNN_NL(nn.Module):
    """Args:
    bn_channels (int): number of channels used in conv_bn_relu layers
    depth (int): number of conv_bn_relu layers
    """
    def __init__(self, in_channels=1, bn_channels=64, depth=18):
        super(DnCNN_NL, self).__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(
                in_channels, bn_channels,
                kernel_size=3, padding=1,
            ),
            nn.ReLU(inplace=True)
        )

        self.conv_bn_relu_list1 = nn.ModuleList(
            [conv3x3_bn_relu(bn_channels, bn_channels) for _ in range(depth//2)]
        )

        self.lr = common.NonLocalLayer_dilation(bn_channels, bn_channels//2, 5, dilation=3, padding=6, ranks=bn_channels//16)

        self.conv_bn_relu_list2 = nn.ModuleList(
            [conv3x3_bn_relu(bn_channels, bn_channels) for _ in range(depth//2)]
        )
        
        self.conv = nn.Conv2d(
            bn_channels, in_channels,
            kernel_size=3, padding=1,
        )

    def forward(self, X):
        residual = X
        out = self.conv_relu(X)
        for layer in self.conv_bn_relu_list1:
            out = layer(out)
        out = self.lr(out)
        for layer in self.conv_bn_relu_list2:
            out = layer(out)
        out = self.conv(out)
        out = out + residual
        return out

def _test_model2d(model):
    torch.manual_seed(1)
    V = Variable(torch.randn(16,3,64,64).cuda())
    S = Variable(torch.randn(16,3,64,64).cuda())
    # import ipdb; ipdb.set_trace()
    net = model(1, 64, depth=18).cuda()
    o_ = []
    for v,s in zip(V.split(1,1), S.split(1,1)):
        o_t = net(v)
        o_.append(o_t)
        loss = nn.MSELoss()(o_t, s)
        loss.backward()
        # import ipdb; ipdb.set_trace()
        param = net.conv.weight
        data = param.grad.data
        # import ipdb; ipdb.set_trace()
        print(param.grad.sum())
        # param.grad = Variable(data.new().resize_as_(data).zero_())
        # import ipdb; ipdb.set_trace()
    o = torch.cat(o_, dim=1)
    # print(net)
    # print(o.shape)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    _test_model2d(DnCNN)