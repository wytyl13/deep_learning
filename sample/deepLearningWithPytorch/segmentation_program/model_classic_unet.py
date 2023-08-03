'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-08-01 15:32:02
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-08-01 15:32:02
 * @Description: unet neural network. unet neural is consist of three main block.
 * convolution block, downsample block and upsample block. so we can define these
 * three block to create one complete unet neural network.


 * convolution -> downsample -> convolution -> downsample -> bottom.
 * bottom -> convolution -> upsample -> contact -> convolution -> upsample -> top.
 * convolution block
        first convolution: in channels != outchannels. the other remain original.
        second convolution: inchannels = outchannels. the other remain original.
    downsample block:
        channels remain the original.
        size scale to 1/2.
    upsample block:
        size scale to 2 times. you can use transpose convolution or linear interpolation.
        channels reduce to 1/2.
 * and you should contact the result of upsample with the correspond location feature_map
    result when you do downsample.
***********************************************************************'''
import torch.nn as nn
import torch
from torch.nn import functional as F


class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            # notice the padding_mode param is used the reflect method to padding. not zero padding
            # notice the bias=False means we will batchnormalize.
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            # the second conv is similar to the first conv.
            # the difference is the input channels.
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )


    def forward(self, x):
        return self.layer(x)
    



'''
 * @Author: weiyutao
 * @Date: 2023-08-01 16:04:04
 * @Parameters: 
 * @Return: 
 * @Description: the orginal paper is used convolution and maxpool to downsample. but the maxpool will drop some
 * feature, so we will use convolution to downsample. you just need to define the kernelsize=3, stride=2, padding=1
 * from inchannels to outchannels. you can reach the similar efficient with the convolution and maxpool.
 '''
class DownSample(nn.Module):

    def __init__(self, channels) -> None:
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU()
        )

    
    def forward(self, x):
        return self.layer(x)




# we can use the transpose convolution or interpolation method.
# reduce the channels and remain the original size.
# upsample will reduce the channels and the size.
# scale to 1/2 size, and reduce to 1/2 channels.
class UpSample(nn.Module):
    def __init__(self, channels) -> None:
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channels, channels // 2, 1, 1)
    

    def forward(self, x, feature_map):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.layer(x)
        # why dim = 1?  batch*channels*m*n
        # the contact will increase the dimension channels. not the size and batch.
        return torch.cat((x, feature_map), dim = 1)




class UNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super(UNet, self).__init__()
        self.conv1 = Conv_Block(3, 64)
        self.d1 = DownSample(64)
        self.conv2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.conv3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.conv4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.conv5 = Conv_Block(512, 1024)
        self.u1 = UpSample(1024)
        self.conv6 = Conv_Block(1024, 512)
        self.u2 = UpSample(512)
        self.conv7 = Conv_Block(512, 256)
        self.u3 = UpSample(256)
        self.conv8 = Conv_Block(256, 128)
        self.u4 = UpSample(128)
        self.conv9 = Conv_Block(128, 64)

        # notice we have created the unet what dedicated to implementing
        # pixel to pixel. if you want to generate one gray value, then, you can
        # change the channel from 3 to 1. or you should use 3 channels to show
        # the color image. and you should notice, the channels is based on the 
        # num calsses. so we should consider the input channel is 64, the output channel
        # is the class number, not the color channels number. and once we 
        # used the num class as  the output channel. we can drop  the sigmoid.
        # because the output channel will give the probability directly.
        self.out = nn.Conv2d(64, num_classes, 3, 1, 1)
        # self.th = nn.Sigmoid()
    


    def forward(self, x):
        R1 = self.conv1(x)
        R2 = self.conv2(self.d1(R1))
        R3 = self.conv3(self.d2(R2))
        R4 = self.conv4(self.d3(R3))
        R5 = self.conv5(self.d4(R4))
        O1 = self.conv6(self.u1(R5, R4))
        O2 = self.conv7(self.u2(R4, R3))
        O3 = self.conv8(self.u3(R3, R2))
        O4 = self.conv9(self.u4(R2, R1))

        return self.out(O4)




if __name__ == "__main__":

    net = UNet()
    input = torch.randn(2, 3, 256, 256)
    # pixel to pixel, 2*3*256*256 to 2*3*256*256
    print(net(input).shape)









