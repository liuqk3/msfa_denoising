from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
# import torch_semiring_einsum
from opt_einsum import contract

class feature_extraction(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(feature_extraction, self).__init__()
        self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv7 = nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=True)
    
    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        return torch.cat((x3,x5,x7), dim=1)

def band_padding(x, adj_ch):
    x_before = x[:,:adj_ch//2,:,:]
    x_after = x[:,-(adj_ch//2):,:,:]
    return torch.cat((torch.flip(x_before, dims=[1]), x, torch.flip(x_after, dims=[1])), dim=1)

class HSIDCNN(nn.Module):
    def __init__(self, adj_ch):
        super(HSIDCNN, self).__init__()
        self.adj_ch = adj_ch

        n_feat = 60
        self.FE_spatial = feature_extraction(1, n_feat//3)
        self.FE_spectral = feature_extraction(adj_ch-1, n_feat//3)

        self.FR1 = nn.Sequential(
            nn.Conv2d(2*n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True))

        self.FR2 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True))
        
        self.FR3 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True))
        
        self.FR4 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True))

        self.REC = nn.Conv2d(4*n_feat, 1, kernel_size=3, stride=1, padding=1, bias=True)
    
    def forward(self, x):
        b,c,h,w = x.size()

        x_pad = band_padding(x, self.adj_ch)
        y = []
        for i in range(c):
            cube = x_pad[:, i:i+self.adj_ch]
            FE_spatial = self.FE_spatial(cube[:,self.adj_ch//2].unsqueeze(1))
            # print(cube[:,:self.adj_ch//2].size(), cube[:,-(self.adj_ch//2):].size())
            FE_spectral = self.FE_spectral(torch.cat((cube[:,:self.adj_ch//2], cube[:,-(self.adj_ch//2):]), dim=1))

            FR1 = self.FR1(torch.cat((FE_spatial, FE_spectral), dim=1))
            FR2 = self.FR2(FR1)
            FR3 = self.FR2(FR2)
            FR4 = self.FR2(FR3)

            REC = self.REC(torch.cat((FR1, FR2, FR3, FR4), dim=1))
            y.append(REC)
        
        return torch.cat(y, dim=1) + x


class feature_extraction_spatial(nn.Module):
    def __init__(self, adj_ch):
        super(feature_extraction_spatial, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, adj_ch, kernel_size=3, stride=1, padding=1, bias=True), nn.LeakyReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(adj_ch, 32, kernel_size=3, stride=1, padding=1, bias=True), nn.LeakyReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, dilation=3, stride=1, padding=3, bias=True), nn.LeakyReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(64, adj_ch, kernel_size=3, dilation=5, stride=1, padding=5, bias=True), nn.LeakyReLU(inplace=True))
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class feature_extraction_spectral(nn.Module):
    def __init__(self):
        super(feature_extraction_spectral, self).__init__()
        k_s = 3
        self.conv1 = nn.Sequential(nn.Conv3d(1, 4, kernel_size=[k_s,3,3], stride=1, padding=[k_s//2,1,1], bias=True), nn.LeakyReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(4, 8, kernel_size=[k_s,3,3], stride=1, padding=[k_s//2,1,1], bias=True), nn.LeakyReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv3d(8, 8, kernel_size=[k_s,3,3], dilation=[1,3,3], stride=1, padding=[k_s//2,3,3], bias=True), nn.LeakyReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv3d(8, 1, kernel_size=[k_s,3,3], dilation=[1,5,5], stride=1, padding=[k_s//2,5,5], bias=True), nn.LeakyReLU(inplace=True))
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class PAM(nn.Module):
    def __init__(self, in_ch):
        super(PAM, self).__init__()
        self.f1 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True), nn.LeakyReLU(inplace=True))
        self.f2 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True), nn.LeakyReLU(inplace=True))
        self.f3 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True), nn.LeakyReLU(inplace=True))
        self.f4 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True), nn.LeakyReLU(inplace=True))
        self.softmax = nn.Softmax(dim=1)
        # self.eq1 = torch_semiring_einsum.compile_equation('bcn,bcm->bnm')
        # self.eq2 = torch_semiring_einsum.compile_equation('bcn,bnm->bcm')

    def forward(self, x):
        b,c,h,w = x.size()
        x1 = self.f1(x)
        x2 = self.f2(x)
        x3 = self.f3(x)
        # a = torch.einsum('bcn, bcm -> bnm', x1.reshape(b,c,-1), x2.reshape(b,c,-1))
        a = contract('bcn, bcm -> bnm', x1.reshape(b,c,-1), x2.reshape(b,c,-1))
        # a = torch_semiring_einsum.einsum(self.eq1, x1.view(b,c,-1), x2.view(b,c,-1), block_size=3)
        a = self.softmax(a)
        # out = torch.einsum('bcn, bnm -> bcm', x3.view(b,c,-1), a)
        out = contract('bcn, bnm -> bcm', x3.view(b,c,-1), a)
        # out = torch_semiring_einsum.einsum(self.eq2, x3.view(b,c,-1), a, block_size=3)
        out = out.view(b,c,h,w)
        out = self.f4(out)+x

        return out

class CAM(nn.Module):
    def __init__(self, in_ch):
        super(CAM, self).__init__()
        self.f1 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True), nn.LeakyReLU(inplace=True))
        self.f2 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True), nn.LeakyReLU(inplace=True))
        self.f3 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True), nn.LeakyReLU(inplace=True))
        self.f4 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True), nn.LeakyReLU(inplace=True))
        self.softmax = nn.Softmax(dim=1)

        # self.eq1 = torch_semiring_einsum.compile_equation('bcn,bsn->bcs')
        # self.eq2 = torch_semiring_einsum.compile_equation('bcn,bcs->bsn')

    def forward(self, x):
        b,c,h,w = x.size()
        x1 = self.f1(x)
        x2 = self.f2(x)
        x3 = self.f3(x)
        # a = torch.einsum('bcn, bsn -> bcs', x1.view(b,c,-1), x2.view(b,c,-1))
        a = contract('bcn, bsn -> bcs', x1.view(b,c,-1), x2.view(b,c,-1))
        # a = torch_semiring_einsum.einsum(self.eq1, x1.view(b,c,-1), x2.view(b,c,-1), block_size=3)
        a = self.softmax(a)
        # out = torch.einsum('bcn, bcs -> bsn', x3.view(b,c,-1), a)
        out = contract('bcn, bcs -> bsn', x3.view(b,c,-1), a)
        # out = torch_semiring_einsum.einsum(self.eq2, x3.view(b,c,-1), a, block_size=3)
        out = out.view(b,c,h,w)
        out = self.f4(out)+x
        
        return out

class SSFE(nn.Module):
    def __init__(self, in_ch):
        super(SSFE, self).__init__()
        k_s = 3
        self.FE = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=[k_s, 5, 5], stride=1, padding=[k_s//2, 2, 2], bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(8, 8, kernel_size=[k_s, 5, 5], stride=1, padding=[k_s//2, 2, 2], bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(8, 1, kernel_size=[k_s, 5, 5], stride=1, padding=[k_s//2, 2, 2], bias=True),
            nn.LeakyReLU(inplace=True))
        self.FUSE1 = nn.Conv2d(2*in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True)

        self.f1 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=True), nn.LeakyReLU(inplace=True))
        self.f2 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, dilation=3, padding=3, bias=True), nn.LeakyReLU(inplace=True))
        self.f3 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, dilation=5, padding=5, bias=True), nn.LeakyReLU(inplace=True))
        self.f4 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, dilation=7, padding=7, bias=True), nn.LeakyReLU(inplace=True))
        self.FUSE2 = nn.Conv2d(4*in_ch, 1, kernel_size=1, stride=1, padding=0, bias=True)
    
    def forward(self, x):
        FE = self.FE(x.unsqueeze(1))[:,0]
        FUSE = self.FUSE1(torch.cat([x, FE], dim=1))

        F1 = self.f1(FUSE)
        F2 = self.f2(FUSE)
        F3 = self.f3(FUSE)
        F4 = self.f4(FUSE)
        out = self.FUSE2(torch.cat([F1, F2, F3, F4], dim=1))

        return out

class HSIDAN(nn.Module):
    def __init__(self, adj_ch):
        super(HSIDAN, self).__init__()
        self.adj_ch = adj_ch

        self.FE_spatial = feature_extraction_spatial(adj_ch)
        self.FE_spectral = feature_extraction_spectral()
        self.PAM = PAM(adj_ch)
        self.CAM = CAM(adj_ch)
        self.SSFE = SSFE(adj_ch)


    def forward(self, x):
        b,c,h,w = x.size()

        x_pad = band_padding(x, self.adj_ch)
        y = []
        for i in range(c):
            cube = x_pad[:, i:i+self.adj_ch]
            FE_spatial = self.FE_spatial(cube[:,self.adj_ch//2].unsqueeze(1))
            # print(cube[:,:self.adj_ch//2].size(), cube[:,-(self.adj_ch//2):].size())
            FE_spectral = self.FE_spectral(cube.unsqueeze(1))[:,0]
            FE_spatial = self.PAM(FE_spatial)
            FE_spectral = self.CAM(FE_spectral)
            FE = FE_spatial + FE_spectral
            REC = self.SSFE(FE)

            y.append(REC)
        
        return torch.cat(y, dim=1) + x




if __name__ == '__main__':
    input = torch.rand((1, 25, 32, 32)).cuda()
    model = HSIDAN(5).cuda()
    model(input)