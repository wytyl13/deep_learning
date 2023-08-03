import torch.nn as nn
import numpy as np
import torch


if __name__ == "__main__":
    # stride = 1, and padding = 1, and kernel = 3
    # the output size is equavalent to input size
    # stride = 2 and padding = 1, and kernel = 3
    # the output size is equavalent to input size / 2.
    # so padding = 1, kernel = 3. if you want to downsample, you should set stride > 1.
    # if you want to remain the original size, you should set stride = 1.
    # and if the kernel size is 7, you should define the padding used 3.
    # so it means the simplest method to downsample the input.
    # kernel=3, stride=2, padding=1, downsample to 1/2 dimension.
    # kernel=7, stride=2, padding=3, downsample to 1/2 dimension.
    # maxpool, kernel=3, stride=2, padding=1, downsample to 1/2 dimension.
    # maxpool kernel=7, stride=2, padding=3, downsample to 1/2 dimension.
    # maxpool, you can define the ceil_mode used true to achieve the same effect.
    # but it is just suitable for the maxpool kernel size is 3. it is not suitable for
    # the maxpool size is 7 or greater.
    # so you should notice, no matter the convolution or maxpool, 
    # if you want to downsample the size based on the stride, you should define the padding.
    # padding = abs(kernel size / 2), just like 7/2 = 3. 3 / 2 = 1.
    # if you want to remain the original size, you should define the stride used zero.
    # and you should notice, if the maxpool kernel size is 2, then the stride is 1.
    # padding=0, you will downsample to 1/2 dimension.

    image = torch.ones((12, 12), dtype=torch.float32)
    image = torch.reshape(image, (1, 1, 12, 12))
    conv_tensor = nn.Conv2d(1, 3, 3, 1, 1)
    maxpool = nn.MaxPool2d(2)
    print(maxpool(image))

