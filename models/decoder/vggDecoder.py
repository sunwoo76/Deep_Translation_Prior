import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class Upconv(nn.Module):
    def __init__(self, cin, cout, k, s, p, norm=False, upsample=False, atv="ReLU", pmode="ReflectionPad2d"):
        super().__init__()
        self.pmode = pmode
        self.upsample = upsample
        
        if self.pmode=="ReflectionPad2d":
            self.padding = nn.ReflectionPad2d((p,p,p,p))
            self.conv = nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=0)
        else:
            self.conv = nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p)

        if norm:
            self.norm = nn.InstanceNorm2d(cout, affine=True)
        else:
            self.norm = None

        if atv=="ReLU":
            self.atv = nn.ReLU()
        elif atv=="LeakyReLU":
            self.atv = nn.LeakyReLU(0.2)
        elif atv=="Sigmoid":
            self.atv = nn.Sigmoid()
        elif atv=="None":
            self.atv = None
        else:
            print("!!!!!!!!! activation option should be defined !!!!!!!!!!!!!! ")
            exit()

        if self.upsample:
            self.upLayer = nn.Upsample(scale_factor=2, mode="nearest")

        self.mapping = None
        if cin != cout:
            self.mapping = nn.Conv2d(cin, cout, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        skip = x
        if self.mapping is not None:
            if self.upsample:
                skip = self.upLayer(skip)
            skip = self.mapping(skip)

        """ upsampling """
        if self.upsample:
            x = self.upLayer(x)

        """ conv, norm , actv """
        if self.pmode=="ReflectionPad2d":
            x = self.padding(x)
        x = self.conv(x)

        if self.norm is not None:
            x = self.norm(x)
        if self.atv is not None:
            x = self.atv(x)

        x = x + skip
        return x
        

class vggDecoder(nn.Module):
    def __init__(self, args, pretrained=False, arch='vgg19', pmode='ReflectionPad2d'):
        super().__init__()
        self.args= args

        self.decoder_blocks_0 = nn.ModuleList([
            # upsample from 64x64
            Upconv(256, 256, 3, 1, 1, norm=True, pmode="RefelectionPad2d"),
            Upconv(256, 256, 3, 1, 1, norm=True, pmode="RefelectionPad2d"),
            Upconv(256, 128, 3, 1, 1, norm=True, pmode="RefelectionPad2d", upsample=True),
        ])

        self.decoder_blocks_1 = nn.ModuleList([
            Upconv(128, 128, 3, 1, 1, norm=True, pmode="RefelectionPad2d"),
            Upconv(128, 128, 3, 1, 1, norm=True, pmode="RefelectionPad2d"),
            Upconv(128, 128, 3, 1, 1, norm=True, pmode="RefelectionPad2d"),
            Upconv(128, 64, 3, 1, 1,  norm=True, pmode="RefelectionPad2d",upsample=True),
        ])

        self.decoder_blocks_2 = nn.ModuleList([
            Upconv(64, 64, 3, 1, 1, norm=True, pmode="RefelectionPad2d"),
            Upconv(64, 64, 3, 1, 1, norm=True, pmode="RefelectionPad2d"),
            Upconv(64, 32, 3, 1, 1, norm=True, pmode="RefelectionPad2d"),
        ])

        self.decoder_blocks_3 = nn.ModuleList([
            Upconv(32, 32, 3, 1, 1, norm=True, pmode="RefelectionPad2d"),
            Upconv(32, 32, 3, 1, 1, norm=True, pmode="RefelectionPad2d"),
            Upconv(32, 3, 3, 1, 1,             pmode="RefelectionPad2d", atv="None"),
        ])
    
    def forward(self, x, warped_feats=None):
        #x = torch.randn( 1,256,64,64, device="cuda").type(torch.cuda.FloatTensor)
        """ single scale case. from 64x64 """
        for f in self.decoder_blocks_0:
            x = f(x)

        for f in self.decoder_blocks_1:
            x = f(x)

        for f in self.decoder_blocks_2:
            x = f(x)

        for f in self.decoder_blocks_3:
            x = f(x)

        return x