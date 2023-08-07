'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-08-03 19:21:14
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-08-03 19:21:14
 * @Description: yolo is one stage working, it is different from
 * maskRcnn and faterRcnn.
 * some indicators for object detecting.
 * map: the composite indicator.
 * precision = TP / (TP + FP)
 * recall = TP / (TP + FN)
 * in our working, we should define one degree of confidence.
 * when greater or equal this degree, we will show it as positive sample.
 * then, the different degree will result to different recall and precision.
 * just like three predict image. the degree is 0.9, 0.8 and 0.7
 * then if you define the degree used 0.9, then the second image and
 * third image result will be dropped. just like the second image has one 
 * positve obejct not be recognized. third image also has one. 
 * so if degree is 0.9, TP will be 1, FP is 0. FN is 2. then 
 * recall = TP / (TP + FN) = 1/3, precision = TP / (TP + FP) = 1.
 * so the positive object that has not be recognized will influence the recall.
 * and the negative sample that be recognized as positive will
 * influence the precision. of course, TP will also influence these two indicators.
 * we can draw one line figure used recall as the horizontal axis and the precision
 * as the vertical aixs. we all known that recall and precision is negative.
 * bigger precision will result the lower recall. and higher recall will result
 * smaller precision. so then, we can calculate the area that this line plot and
 * horizontal and vertical involved. this area is MAP indicator.
 * the biggest area is 1 for MAP. it means as the recall increased, 
 * the precision remaind as 1 value. so the area will be 1.
 *


 * yolov1, inpput size is 448*448*3, can not change it. because you should
 * corresponding it to full connection.
 * 448*448*4 --> all convolution --> 7*7*1024, passed it into full connection.
 * 7*7*1024 -> full connection, 4096*1 --> full connection, 1470*1 --> reshape
 * to 7*7*30, 7*7 is the grid size, 30 is equal to two bound box + 20 classes.
 * x1, y1, w1, h1, c1, x2, y2, w2, h2, c2, classes=20. this size is 30
 * has two bounding boxes. because this construct will show all the detected
 * object in one image, so yolo is one stage algorithm.
 * then the most important working is to design the loss function:
    location error: 
        x, y, w, h. error.(x均方误差)+(y均方误差)+(根号下w均方误差)+(根号下h均方误差)
    degree of confidence error:
        confidence均方误差, the difference between the real iou and predict confidence.
        you should distingush the object confidence error and nonobject confidence error.
        why consider the nonobject and object? because one image might has amount
        background what means nonobject, so the nonobject confidence error will be greater
        than the object confidence error. so we should use one 0 to 1 value to scale
        the nonobject confidence error, so much so that it will not influence our error too much.
        another reason is the object is more important than the background.
        so we should scale the influence by the nononject confidence error.
    classifier error:
        crossentropy loss function. the actual probability and predict probability.
 * that's all content about yolov1, but the disadvantages is it is not friendly
 * for small object and overlap object.
 *

 * yolov2 dropped the drop out layer, used batch normalization to instead.
 * batch normalization after each convolution. avoid to the gradient disappear.
 * batch normalization increased MAP to 2%.
 * yolov2 used 448*448 to train the model. v1 used 224*224 to train model.
 * this increase the MAP to 4%.
 * add the darknet. drop the full connection, used convolution to instead.
 * save many parameters. five times maxpool to reduce the size. 
 * yolov1:
    input 224*224 to 7*7, reduce 32 = 2^5.
 * yolov2:
    input 416*416 to 13*13, reduce 32 = 2^5. why 416, because we want to get 
    one odd number just like 3, 7, 11, 13 and so on.
    the bigger output size 13*13 will result to better detected. V1 is 7*7
    V2 is 13*13.
 * darknet has 19 convolution. 5 maxpool.
 * the anchor boxes in yolov1 is 2, yolov2 is 5. 5 is calculated based on 
 * k-mean algorithm what based on the IOU distance not euclident distance.
 * but problem is the more achor boxed has not result to the bigger MAP value.
 * but it result to bigger recall. from 81% increased to 88%.



 * receptive field.
 * just like input is h*w*c, we want to generate c feature mapping.
 * we will compare the parameters numbers about one 7*7 convolusion 
 * and three 3*3 convolution
 * we should use c*(7*7*c) = 49c^2 parameters if we used 7*7 convolution kernel.
 * we should use 3*c*(3*3*c) = 27c^2 if we used three 3*3 convolution kernel.
 * and smaller kernel is suitable for feature extraction.
 * because dropped the full connection, so yolov2 can detect different size image.
 * 320*320 to 608*608, but the premise condition is can be divided exactly by 32.
 * because we will do five times maxpool.


 * yolov3, more suitable for small object. achor boxse from 5 to 9.
 * update the softmax, can predict multiple tags task. yolov2 add the 
 * first layer result to the last layer to avoid ignore the small object.
 * because the receptive field of last convolution layer is large, it is not
 * friendly for small object. but this method is not also suitable for the big
 * object detected. so yolov3 changed it. yolov3 designed three scale feature.
 * yolov1 is 7*7, yolov2 is 13*13, but yolov3 has three scale feature mapping.
 * is 13*13, 26*26 and 52*52, the first is dedicated to the biggest object detected.
 * the middle is dedicated to the middle object size, the last is dedicated to
 * the smallest object. and each scale has three achor boxes. so there are
 * 9 achor boxes. and you should notice these three scale merging each. because
 * the last smallest size 13*13 can get biggest information for the original image.
 * so the bigger scale can get some useful information for original image from
 * the smaller scale.
 * then the core problem is how to fusion these three scale. upsample or interpolation.
 * 13*13 upsample to 26*26. and then 26*26 sample to 52*52. so it is similar to
 * unet, and you should notice the last layer of yolov2 also used the the concept of resnet.

 * then we started to learn the darknet53 network of yolov3.
 * dropped the maxpool and full connection, used the convolution. set stride as 2
 * to instead the maxpool.
 * notice what is downsample? reduce the image size.
 * upsample is to increase the image size.
 * the backbone network for feature extraction for yolov3, we also named it as darknet53:
 * input 416*416*3
 * conv1 3*3 kernel(3, 32, stride=1, padding=1) -> (416*416*32)
 * conv2 and downsample 3*3 kernel (32, 64, stride=2, padding=1) -> (208*208*64)
 * residual block1 * 1:
        input: (208*208*64)
        convolution and not downsample. remain the input size.
            conv3 1*1 kernel(64, 32, stride=1, padding=0) -> (208, 208, 32)
            conv4 3*3 kernel(32, 64, stride=1, padding=1) -> (208, 208, 64)
            residual: input + the former out. -> (208, 208, 64)
    conv5, downsample 3*3 kernel(64, 128, stride=2, padding=1) -> (104, 104, 128)
 * residual block2 * 2:
        input: (104, 104, 128)
        convolution and dowsample used convolution. remain the input size.
            conv6 1*1 kernel(128, 64, stride=1, padding=0) -> (104*104*64)
            conv7 3*3 kernel(64, 128, stride=1, padding=1) -> (104, 104, 128)
            residual: input + the former conv7 result = (104, 104, 128)
    conv8, downsample 3*3 kernel(128, 256, stride=2, padding=1) -> (52, 52, 256)
 * residual block3 * 8:
        input: (52, 52, 256)
        convolution and not downsample. remain the inputsize.
            conv9 1*1 kernel(256, 128, stride=1, padding=0) -> (52, 52, 128)
            conv10 3*3 kernel(128, 125, stride=1, padding=1) -> (52, 52, 256)
            residual: input + the former out conv10. -> (52, 52, 256)
    conv11, downsample 3*3 kernel(126, 512, stride=2, padding=1) -> (26, 26, 512)
 * residual block4 * 8
        input: (26, 26, 512)
        convolution and not downsample, remain the inputsize.
            conv12: 1*1 kernel(512, 256, stride=1, padding=0) -> (26, 26, 256)
            conv13: 3*3 kernel(256, 512, stride=1, padding=1) -> (26, 26, 512)
            residual: input + ocnv13 out. -> (26, 26, 512)
 * conv14, downsample 3*3 kernel(512, 1024, stride=2, padding=1) -> (13, 13, 1024)
 * residual block5 * 4:
        input: (13, 13, 1024)
        convolution and not downsample, remain the input size.
            conv15: 1*1 kernel(1024, 512, stride=1, padding=0) -> (13, 13, 512)
            conv16: 3*3 kernel(512, 1024, stride=1, padding=1) -> (13, 13, 1024)
            residual: input + conv16 result. -> (13, 13, 1024)
 * the above structure is darknet53.
 * all layer numbers is equal to 
 * 13*13*85, 85 = (80 classes + x + y + w + h + c)
 * notice 1*1 convolution kernel has not the feature extraction function.
 * it is generally used for reducing the channel numbers. so we can use 3*3 to implement
 * both feature extraction and recude the channel. and we can use the nearest interpolation
 * method to do upsample. of course we can also use the linear interpolation, but the nearest
 * will not add the new feature value. notice we will define the last part for the 
 * yolov3 model. the last part we also consider the former feature mapping into the current
 * feature mapping, but we do not use the residual method. notice you should known the 
 * difference between residual and record concat, the former is math computation and
 * the last is physical joining together. we have used the physical joining during the
 * upsampling operation in unet model. just like the code as follow.
    return torch.cat((x, feature_map), dim = 1)
 * its efficient is different from the residual layer.
 * just like the difference between 1+1=2 and [1, 1], the last can remain more information
 * about the orginal information. 
 * then when we should use the cat operation? when the depth is different.
 * if the different is large, you should use cat not the residual. of course, the cat 
 * operation is after the upsample. because it is dedicated to make up for the loss that we
 * have dropped some feature information during the upsampling. then, we can give the last
 * part model after the darknet53 net in yolov3.
 * the last part model.
 * input: (13, 13, 1024)
 * convolution set(13):
        conv17: 1*1 kernel(1024, 512, stride=1, padding=0) -> (13, 13, 512)
        conv18: 3*3 kernel(512, 1024, stride=1, padding=1) -> (13, 13, 1024)
        conv19: 1*1 kernel(1024, 512, stride=1, padding=0) -> (13, 13, 512)
        conv20: 3*3 kernel(512, 1024, stride=1, padding=1) -> (13, 13, 1024)
        conv21: 1*1 kernel(1024, 512, stride=1, padding=0) -> (13, 13, 512)
 * then we will divide into two road.
 * first road for convolutionSet13:
        use 13*13 to predict the biggest object.
        input: (13, 13, 512)
        conv22: 3*3 kernel(512, 1024, stride=1, padding=1) -> (13, 13, 1024)
        conv23: 1*1 kernel(1024, 3*(5+num_classes), stride=1, padding=0) -> (13, 13, 3*(5+num_classes))
 * second road for convolutionSet13:
        input: (13, 13, 512)
        use 13*13 to help the 26*26 predict. it means we will operate upsample and cat.
        conv24: 3*3 kernel(512, 256, stride=1, padding=1) -> (13, 13, 256)
        upsample: nearest interpolation -> (26, 26, 256)
        cat with darknet53_26, torch.cat((26, 26, 256), (26, 26, 512), dim=1) => (26, 26, 768)
 * convolution set(26):
        input: (26, 26, 768)
        output: (26, 26, 256)
 * first road for convolutionSet26:
        to predict used 26
        conv25: 3*3 kernel(256, 512, stride=1, padding=1) -> (26, 26, 512)
        conv26: 1*1 kernel(512, 3*(5+sum_class), stride=1, padding=0)
 * second road for convolutionSet26:
        input: (26, 26, 256)
        conv27: 3*3 kernel(256, 128, stride=1, padding=1) -> (26, 26, 128)
        upsample: nearest interpolation -> (52, 52, 128)
        cat with darknet53_52, torch.cat((52, 52, 128), (52, 52, 256), dim=1) -> (52, 52, 384)
 * convolution set(52):
        input(52, 52, 384)
        output(52, 52, 128)
 * just one road for convolutionSet52:
        predict used 52
        conv28: 3*3 kernel(128, 256, stride=1, padding=1) -> (52, 52, 128)
        conv29: 1*1 kernel(128, 3*(5+num_class), stride=1, padding=0)
 * and so on. we will code the last part at last. then we will start to generate our code
 * for yolov3. we will generate three block:
        one is convolution block what involved convolution, batchnormalization and sigmoid 
        function.
        one is residual block what involved convolution, residual.
        one is convolution set what is the last part model after darknet53. 
 * notice, if we want to downsample what means we want to reduce the size, we should use 3*3 kernel
 * and set stride=2, padding=1. or we can use maxpool. but maxpool will drop the key value.
 * if we want to reduce the channels, we should use 1*1 kernel, and stride=1, padding=0 to convolution.
 * generally, the bigger size for feature mapping, the large number feature mapping.
 * then, we will create the reading data code and train the data used yolov3 model what we have created successful.
***********************************************************************'''

import torch.nn as nn
import torch.nn.functional as F
import torch
import config



'''
 * @Author: weiyutao
 * @Date: 2023-08-04 11:00:23
 * @Parameters: 
 * @Return: 
 * @Description: define the convolution block.
 '''
class ConvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False) -> None:
        super(ConvolutionLayer, self).__init__()
        self.sub_model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.sub_model(x)



'''
 * @Author: weiyutao
 * @Date: 2023-08-04 11:07:02
 * @Parameters: 
 * @Return: 
 * @Description: define the residual block.
 '''
class ResidualLayer(nn.Module):
    def __init__(self, in_channels) -> None:
        super(ResidualLayer, self).__init__()
        self.sub_model = nn.Sequential(
            ConvolutionLayer(in_channels, in_channels // 2, 1, 1, 0),
            ConvolutionLayer(in_channels // 2, in_channels, 3, 1, 1)
        )


    def forward(self, x):
        return x + self.sub_model(x)
    


'''
 * @Author: weiyutao
 * @Date: 2023-08-04 11:08:03
 * @Parameters: 
 * @Return: 
 * @Description: define the convolution set layer.
 '''
class ConvolutionSetLayer(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(ConvolutionSetLayer, self).__init__()
        self.sub_model = nn.Sequential(
            ConvolutionLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionLayer(out_channels, in_channels, 3, 1, 1),
            ConvolutionLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionLayer(out_channels, in_channels, 3, 1, 1),
            ConvolutionLayer(in_channels, out_channels, 1, 1, 0)
        )


    def forward(self, x):
        return self.sub_model(x)


'''
 * @Author: weiyutao
 * @Date: 2023-08-04 11:11:48
 * @Parameters: 
 * @Return: 
 * @Description: define the downsampling block
 '''
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(DownSample, self).__init__()
        self.sub_model = nn.Sequential(
            ConvolutionLayer(in_channels, out_channels, 3, 2, 1)
        )


    def forward(self, x):
        return self.sub_model(x)
    

'''
 * @Author: weiyutao
 * @Date: 2023-08-04 11:13:56
 * @Parameters: 
 * @Return: 
 * @Description: define the ipsampling block
 '''
class UpSample(nn.Module):
    def __init__(self) -> None:
        super(UpSample, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')




'''
 * @Author: weiyutao
 * @Date: 2023-08-04 11:17:06
 * @Parameters: 
 * @Return: 
 * @Description: define the yolov3 model.
 '''
class Yolov3(nn.Module):
    def __init__(self) -> None:
        super(Yolov3, self).__init__()
        self.darknet53_52 = nn.Sequential(
            ConvolutionLayer(3, 32, 3, 1, 1),
            ConvolutionLayer(32, 64, 3, 2, 1),

            # residual block1
            ResidualLayer(64),

            DownSample(64, 128),

            # residual block2
            # remain the input channel. down and up the channel.
            ResidualLayer(128),
            ResidualLayer(128),

            DownSample(128, 256),

            # residual block3
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256)
        )

        self.darknet53_26 = nn.Sequential(

            DownSample(256, 512),

            # residual block4
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512)
        )

        self.darknet53_13 = nn.Sequential(

            DownSample(512, 1024),

            # residual block5
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024)
        )

        # convolutionset layer will reduce the channels to 512. remain the input size.
        self.convolutionSet13 = nn.Sequential(
            # input: 13, 13, 1024
            ConvolutionSetLayer(1024, 512)
        )

        # then divided into two road, one to predict, one to upsample to help the bigger size.
        self.detect_13 = nn.Sequential(
            # convolution two.
            # input (13*13*512)
            # the first convolution will increase the channel.
            ConvolutionLayer(512, 1024, 3, 1, 1),
            # the second convolution will predict the result.
            # the out_channels depend on the result what you want.
            # yolov3 has three scale achor boxes, each achor boxes has 15 element(x, y, w, h, c) + classes number.
            # just like we want to classifier 10 class, so the out_channels is equal to (5+10)*3 = 45
            nn.Conv2d(1024, 3 * (config.CLASS_NUM + 5), 1, 1, 0)
        )

        # the second road. convolution, upsample, convolution.
        self.up_to_26 = nn.Sequential(
            # input: (13, 13, 512)
            # 1*1 kernel size to convolution, reduce the channels.
            ConvolutionLayer(512, 256, 3, 1, 1),
            # upsample to (26, 26, 256)
            UpSample()
        )

        self.convolutionSet26 = nn.Sequential(
            # input: (26, 26, 256+512)
            # because cat((26, 26, 256), (26, 26, 512)) before this function.
            # this input channel is after catting. 256+512=768
            ConvolutionSetLayer(768, 256)
        )
        # the first road predict 26*26
        self.detect_26 = nn.Sequential(
            ConvolutionLayer(256, 512, 3, 1, 1),
            nn.Conv2d(512, 3 * (config.CLASS_NUM + 5), 1, 1, 0)
        )


        self.up_to_52 = nn.Sequential(
            # input: (26, 26, 256) is the output of convolutionSet26
            ConvolutionLayer(256, 128, 3, 1, 1),
            # upsample to (52, 52, 128)
            UpSample()
        )

        self.convolutionSet52 = nn.Sequential(
            # input: (52, 52, 128+256)
            # cat((52, 52, 128), (52, 52, 256))
            ConvolutionSetLayer(384, 128)
        )
        
        # 52 size just has one road.
        self.detect_52 = nn.Sequential(
            ConvolutionLayer(128, 256, 3, 1, 1),
            nn.Conv2d(256, 3 * (config.CLASS_NUM + 5), 1, 1, 0)
        )


    # notice we can not add any symbol after one code, it will pass into the function as param.
    # so it will be error.
    def forward(self, x):
        dark53_52 = self.darknet53_52(x)
        dark53_26 = self.darknet53_26(dark53_52)
        dark53_13 = self.darknet53_13(dark53_26)
        # dark52 is the input of convolutionSet13.
        convolutionSet13 = self.convolutionSet13(dark53_13)
        # the first road to predict 13 for convolutionSet13
        detecttion_13 = self.detect_13(convolutionSet13)
        # the second road to help 25 for convolutionSet13 what invloved upsample and cat with dark53_26
        # because the channel at the index 1, (batch, channel, width, height), so we should cat based on dim=1.
        up_to_26 = self.up_to_26(convolutionSet13)
        cat_26 = torch.cat((up_to_26, dark53_26), dim=1)

        # input the cat_26 into function convolutionSet26
        convolutionSet26 = self.convolutionSet26(cat_26)
        # two road. first is predict, second is upsample and help 52.
        detection_26 = self.detect_26(convolutionSet26)
        up_to_52 = self.up_to_52(convolutionSet26)
        cat_52 = torch.cat((up_to_52, dark53_52), dim=1)

        # input the cat_52 into functon convolutionSet26. then we just has one road to predict.
        convolutionSet52 = self.convolutionSet52(cat_52)
        detection_52 = self.detect_52(convolutionSet52)

        return detecttion_13, detection_26, detection_52




if __name__ == "__main__":
    net = Yolov3()
    x = torch.randn(1, 3, 416, 416)
    out = net(x)
    print(out[0].shape, out[1].shape, out[2].shape)










