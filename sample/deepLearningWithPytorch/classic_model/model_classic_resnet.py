'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-08-01 10:32:36
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-08-01 10:32:36
 * @Description: we will create the resnet. the mode is conv+batchnormalization+maxpool
 * We can check the model we have created used timm
 * import timm
 * timm.create_model("resnet50")
 * resnet: 50, 101, 152
 * input: batch*3*224*224
 * conv1: 7*7, 64, stride=2, batch*64*112*112
 * maxpool: 3*3, stride=2, batch*64*56*56
 * block1, stride=1, 56*56: 
 *      1*1, 64
        3*3, 64
        1*1, 256
   block2, stride=2, 28*28:
        1*1, 128
        3*3, 128
        1*1, 512
   block3, stride=2, 14*14:
        1*1, 256
        3*3, 256
        1*1, 1024
   block4, stride=2, 7*7:
        1*1, 512
        3*3, 512
        1*1, 2048
* average pool:  2048 feature
* linear: fc 1000 classes, softmax
* block1, block2, block3, block4 = 
* resnet50: [3, 4, 6, 3]
* resnet101: [3, 4, 23, 3]
* resnet152: [3, 8, 36, 3]
* we should add the residual at the first circle for each block. just like block1
* block1:
        input x(n*n*64)
        x1 = conv1(64, 64, 1, 1, 0)
        x2 = conv2(64, 64, 3, 1, padding=1)
        x3 = conv3(64, 256, 1, 1, 0)
        x--> conv4(64, 256, 1, 1, 0) = (n, n, 256) + x3(n, n, 256) = x4
        input x4(n, n, 256)
        cycle:
            if stride != 1 or in_channels != out_channels*4
            x5 = conv1(256, 64, 1, 1, 0)

***********************************************************************'''


import numpy as np
import torch
import torch.nn as nn



class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        # store the original x. we will use it at last.
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)

        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes) -> None:
        super(ResNet, self).__init__()
        self.in_channels = 64
        # the first conv1, if you want to downsample to 1/2 size, and kernel size is 7, 
        # you should use 7, 2, 3
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self.make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self.make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self.make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self.make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x





    def make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []
        # if stride != 1 means
        # this code as follow just apply for the first block in one layer.
        # we should define the identity_downsample in first block in each layer.
        # and we should set identity_downsample used None at start.
        # the first block in one layer in_channels != out_channels * 4
        # the other block in one layer in_channels is equavalent to out_channels*4
        # just like the first block in first layer, in_channels = out_channels = 64
        # the second block in first layer, in_channels = out_channel * 4. 
        if stride != 1 or self.in_channels != out_channels * 4:
            # notice, just the first block in current layer will downsample
            # and just the first block used stride = 2. so the first block will be very complex in one layer.
            # the first block in one layer should use the passed stride, the other block in one layer should use one
            # as the stride value. and the residual should downsample from n to 4*n
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride=stride),
                                                nn.BatchNorm2d(out_channels*4))
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels * 4
        for i in range(num_residual_blocks - 1):
            # the other block in one layer should use 1 as the stride value.
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)





def ResNet50(image_channels = 3, num_classes = 1000):
    return ResNet(block, [3, 4, 6, 3], image_channels, num_classes)

def ResNet101(image_channels = 3, num_classes = 1000):
    return ResNet(block, [3, 4, 23, 3], image_channels, num_classes)

def ResNet152(image_channels = 3, num_classes = 1000):
    return ResNet(block, [3, 8, 36, 3], image_channels, num_classes)

def test():
    net = ResNet50()
    x = torch.randn(2, 3, 224, 224)
    if torch.cuda.is_available():
        y = net(x).to('cuda')
    else:
        y = net(x).to('cpu')
    print(y.shape)



if __name__ == "__main__":
    # print: 2*1000
    # test()
    # net = ResNet152()
    # x = torch.randn(2, 3, 224, 224)
    # y = net(x).to('cpu')
    # print(y.shape)

























