import os
import torch.nn as nn
import torch.optim as optim
from base_networks import *
from torchvision.transforms import *

class Net(nn.Module):
    def __init__(self, base_filter, feat, num_stages, scale_factor):
        super(Net, self).__init__()
        
        if scale_factor == 2:
        	kernel = 6
        	stride = 2
        	padding = 2
        elif scale_factor == 4:
        	kernel = 8
        	stride = 4
        	padding = 2
        elif scale_factor == 8:
        	kernel = 12
        	stride = 8
        	padding = 2
        
        #Initial Feature Extraction
        #self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(base_filter, feat, 1, 1, 0, activation='prelu', norm=None)
        #Back-projection stages
        self.up1 = UpBlock(feat, kernel, stride, padding)
        self.down1 = DownBlock(feat, kernel, stride, padding)
        self.up2 = UpBlock(feat, kernel, stride, padding)
        self.down2 = DownBlock(feat, kernel, stride, padding)
        self.up3 = UpBlock(feat, kernel, stride, padding)
        #Reconstruction
        self.output = ConvBlock(num_stages*feat, feat, 1, 1, 0, activation=None, norm=None)
        
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            
    def forward(self, x):
        #x = self.feat0(x)
        x = self.feat1(x)
        
        h1 = self.up1(x)
        h2 = self.up2(self.down1(h1))
        h3 = self.up3(self.down2(h2))
        
        x = self.output(torch.cat((h3, h2, h1),1))
        
        return x
