import torch
import torch.nn.functional as F
from torch import nn
import math
import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
# from netvlad import NetVLADLoupe
import numpy as np
import torch.nn.init as init

from .lifting import LiftingScheme2D
from .PyramidNetVLAD import WPNNetVLAD
from .Swin_MFT import MFTBlock


class ITRNet_P(nn.Module):
    def __init__(self, height=64, width=900, channels=2, dct_kernel=(8,8), num_heads=[2, 4, 8], drop_paths=0.):
        super(ITRNet_P, self).__init__()
        self.dct_kernel = dct_kernel
        self.num_heads = num_heads

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=(5,1), stride=(1,1), bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,1), stride=(2,1), bias=False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,1), stride=(2,1), bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3,1), stride=(2,1), bias=False) #H6
    
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(2,1), stride=(2,1), bias=False) #H3
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False) #H2
        self.conv7 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False) #H1

        self.conv8 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False) #01
        self.conv9 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False) #02
        self.conv10 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False) #03
        self.conv11 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False) #04
        self.relu = nn.ReLU(inplace=True)


        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=1024, activation='relu', batch_first=True,dropout=0.)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.convLast1 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(1,1), bias=False)
        self.convLast2 = nn.Conv2d(256, 512, kernel_size=(1,1), stride=(1,1), bias=False)
        
        self.wavelet1 = LiftingScheme2D(in_planes=256, share_weights=True)
        self.wavelet2 = LiftingScheme2D(in_planes=256, share_weights=True)

        self.mft1 = MFTBlock(head_dim = 16)
        self.mft2 = MFTBlock(head_dim = 8)
        self.mft4 = MFTBlock(head_dim = 4)

        feature_size = [256,256,256]
        cluster_size = [4,16,32]
        output_dim = [1024,1024,1024]
        max_samples = [64,256,1024]
        self.PyramidNetVLAD = WPNNetVLAD(feature_size, max_samples,cluster_size,output_dim)


    def forward(self, x_0):
        
        x_l = F.interpolate(x_0, size=(64,1024), mode='bilinear', align_corners=False)
        out_l = self.relu(self.conv1(x_l))
        out_l = self.relu(self.conv2(out_l))
        out_l = self.relu(self.conv3(out_l))
        out_l = self.relu(self.conv4(out_l))
        out_l = self.relu(self.conv5(out_l))
        out_l = self.relu(self.conv6(out_l))
        out_l = self.relu(self.conv7(out_l))
        out_l = self.relu(self.conv8(out_l))
        out_l = self.relu(self.conv9(out_l))
        out_l = self.relu(self.conv10(out_l))
        out_l = self.relu(self.conv11(out_l))

        out_l_1 = out_l.permute(0,1,3,2)
        out_l_1 = self.relu(self.convLast1(out_l_1))
        
        out_l_2 = out_l_1.squeeze(3) #B C L 
        out_l = out_l_2.permute(2, 0, 1) #L B C 
        out_l = self.transformer_encoder(out_l)#L B C
        out_l = out_l.permute(1, 2, 0)# B C L 

        out_l = torch.cat((out_l, out_l_2), dim=1)
        x_5 = out_l.reshape(out_l.size(0), out_l.size(1), 32, 32)
        _, _, x_5_1, _, _, _ = self.wavelet1(x_5)
        _, _, x_5_2, _, _, _ = self.wavelet2(x_5_1)
        
        x_5 = self.mft1(x_5)
        x_5_1 = self.mft2(x_5_1)
        x_5_2 = self.mft4(x_5_2)

        out_l = F.normalize(x_5, dim=1)
        out_2 = F.normalize(x_5_1, dim=1)
        out_3 = F.normalize(x_5_2, dim=1)
        
        out_l = self.PyramidNetVLAD(out_3, out_2, out_l)
        out_l = F.normalize(out_l, dim=1)

        return out_l
        
        

if __name__ == '__main__':

    combined_tensor = torch.randn(2,2,64,1024).cuda()
    feature_extracter=ITRNet_P()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extracter.to(device)

    gloabal_descriptor = feature_extracter(combined_tensor)
    print("size of gloabal descriptor: \n")
    print(gloabal_descriptor.size())
