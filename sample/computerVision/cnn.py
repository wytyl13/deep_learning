'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-07-14 08:37:49
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-07-14 08:37:49
 * @Description: the convolution neural network is consist of convolution layer, 
 * pooling layer and full connected layer. then the convariance layer and pooling layer
 * can have many times, and the full connected is behind all the convariance layer and pooling
 * layer, the constrcture of cnn model can expressed as follow:
 * input --> convolution layer --> pooling layer --> convolution layer --> pooling layer -->
 * ...... --> full connected layer.
 * full connected layer can expressed as follow:
 * the output of pooling layer --> hidden layer --> output layer.
 * of course, if you want to handle the color image, just like the three channel color image.
 * you can also define one three dimension convolution kernel. then we can generate one 
 * four dimension feature mapping, then for the next round of convolution. 
 * the size of output is as follow, just like the zero padding is P, the convolution 
 * size is F, the input size is W, the step is S, then the output size is equal to
 * (W + 2P - F) / S + 1. the numbers of weights parameters is equal to (C * F * F + 1) * k
 * K is the numbers of convolution kernel, C is the numbers of channel. F is the size of convolution kernel.
 * consider to the bias parameter, add one.
***********************************************************************'''