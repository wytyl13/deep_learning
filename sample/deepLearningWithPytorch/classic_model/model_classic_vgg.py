'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-08-01 14:48:02
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-08-01 14:48:02
 * @Description: vgg16 and vgg19
 * input (224*224*3)
 * conv1*2: 3*3, 224*224*64, maxpool*1: 2*2 -> 112*112*64
 * conv2*2: 3*3, 112*112*128, maxpool*1: 2*2 -> 56*56*128
 * conv3*3: 3*3, 56*56*256, maxpool*1: 2*2 -> 28*28*256
 * conv4*3: 3*3, 28*28*512, maxpool*1: 2*2 -> 14*14*512
 * conv5*3: 3*3, 14*14*512, maxpool*1: 2*2 -> 7*7*512
 * full connection1: 7*7*512 -> 1*1*4096 -> 4096 -> 1000
 * total layers = 2+2+3+3+3+3 = 16


 * we can use the convolution to instead the full connection.
 * just like the full connection is
        7*7*512 -> 4096
    we can define one kernelsize = 7*7, stride = 1, padding = 0 
    and out_channels is 4096 to convolution.
    just like Conv2d(512, 4096, 7, 1, 0)
    we will get the same efficient with the full connection.
    # the output size is (7-7+2*0)/1 = 0
***********************************************************************'''
import torch.nn as nn
import numpy as np
import torch

class vgg16(nn.Module):
    def __init__(self) -> None:
        super(vgg16, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.maxpool = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv7 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv8 = nn.Conv2d(512, 512, 3, 1, 1)
        self.fn1 = nn.Linear(7*7*512, 4096)
        self.fn2 = nn.Linear(4096, 4096)
        self.fn3 = nn.Linear(4096, 1000)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)

        # conv6*2
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv6(x)
        x = self.maxpool(x)

        # conv8 * 2
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv8(x)
        x = self.maxpool(x)

        # conv8 * 3
        x = self.conv8(x)
        x = self.conv8(x)
        x = self.conv8(x)
        x = self.maxpool(x)
        # full connection.
        x = x.reshape(x.shape[0], -1)
        x = self.fn1(x)
        x = self.fn2(x)
        x = self.fn3(x)

        return x
    
if __name__ == "__main__":
    vgg16 = vgg16()
    input = torch.randn(2, 3, 224, 224)
    y = vgg16(input).to('cpu')
    print(y.shape)
