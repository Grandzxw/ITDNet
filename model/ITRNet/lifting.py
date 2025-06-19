# Copyright 2024 - xiongwei zhao @ grandzhaoxw@gmail.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# To change if we do horizontal first inside the LS
HORIZONTAL_FIRST = True

class Splitting(nn.Module):
    def __init__(self, horizontal):
        super(Splitting, self).__init__()
        # Deciding the stride base on the direction
        self.horizontal = horizontal
        if(horizontal):
            self.conv_even = lambda x: x[:, :, :, ::2]
            self.conv_odd = lambda x: x[:, :, :, 1::2]
        else:
            self.conv_even = lambda x: x[:, :, ::2, :]
            self.conv_odd = lambda x: x[:, :, 1::2, :]

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.conv_even(x), self.conv_odd(x))



class LiftingScheme(nn.Module):
    def __init__(self, horizontal, in_planes, modified=True, size=[], splitting=True, k_size=4, simple_lifting=False):
        super(LiftingScheme, self).__init__()
        self.modified = modified
        if horizontal:
            kernel_size = (1, k_size)
            pad = (k_size // 2, k_size - 1 - k_size // 2, 0, 0)
        else:
            kernel_size = (k_size, 1)
            pad = (0, 0, k_size // 2, k_size - 1 - k_size // 2)

        self.splitting = splitting
        self.split = Splitting(horizontal)

        # Dynamic build sequential network
        modules_P = []
        modules_U = []
        prev_size = 1

        # HARD CODED Architecture
        if simple_lifting:            
            modules_P += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]
            modules_U += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]
        else:
            size_hidden = 2
            
            modules_P += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes*prev_size, in_planes*size_hidden,
                          kernel_size=kernel_size, stride=1),
                nn.ReLU()
            ]
            modules_U += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes*prev_size, in_planes*size_hidden,
                          kernel_size=kernel_size, stride=1),
                nn.ReLU()
            ]
            prev_size = size_hidden

            # Final dense
            modules_P += [
                nn.Conv2d(in_planes*prev_size, in_planes,
                          kernel_size=(1, 1), stride=1),
                nn.Tanh()
            ]
            modules_U += [
                nn.Conv2d(in_planes*prev_size, in_planes,
                          kernel_size=(1, 1), stride=1),
                nn.Tanh()
            ]

        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        if self.modified:
            c = x_even + self.U(x_odd)
            d = x_odd - self.P(c)
            return (c, d)
        else:
            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)
            return (c, d)



class LiftingScheme2D(nn.Module):
    def __init__(self, in_planes, share_weights, modified=True, size=[2, 1], kernel_size=4, simple_lifting=False):
        super(LiftingScheme2D, self).__init__()
        self.level1_lf = LiftingScheme(
            horizontal=HORIZONTAL_FIRST, in_planes=in_planes, modified=modified,
            size=size, k_size=kernel_size, simple_lifting=simple_lifting)
        
        self.share_weights = share_weights
        
        ##这里是实例化一个对象
        if share_weights:
            self.level2_1_lf = LiftingScheme(
                horizontal=not HORIZONTAL_FIRST,  in_planes=in_planes, modified=modified,
                size=size, k_size=kernel_size, simple_lifting=simple_lifting)
            self.level2_2_lf = self.level2_1_lf  # Double check this
        else:
            self.level2_1_lf = LiftingScheme(
                horizontal=not HORIZONTAL_FIRST,  in_planes=in_planes, modified=modified,
                size=size, k_size=kernel_size, simple_lifting=simple_lifting)
            self.level2_2_lf = LiftingScheme(
                horizontal=not HORIZONTAL_FIRST,  in_planes=in_planes, modified=modified,
                size=size, k_size=kernel_size, simple_lifting=simple_lifting)

    def forward(self, x):
        '''Returns (LL, LH, HL, HH)'''
        (c, d) = self.level1_lf(x)
        (LL, LH) = self.level2_1_lf(c)
        (HL, HH) = self.level2_2_lf(d)
        return (c, d, LL, LH, HL, HH)




if __name__ == '__main__':

    #[4,3,200,200] 
    wavelet = LiftingScheme2D(in_planes=3, share_weights=True)    
    
    # feature = featureExtracter_RI_BEV_256(channels=12, use_transformer=use_transformer)
    # feature = FeatureExtracter_BEV_XY(height=256)
    #feature = FeatureExtracter_BEV_Z(height=48)
     
    x1 = torch.randn(4, 3, 200, 200)
    c, d, LL, LH, HL, HH = wavelet(x1)
    
    
    print("output shape:", LL.shape)
