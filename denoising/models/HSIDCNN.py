import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

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