#!C:/Users/80521/AppData/Local/Programs/Python/Python38 python
# -*- coding=utf8 -*-
'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-07-04 14:49:39
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-07-04 14:49:39
 * @Description: this file we will learn the chapter 1 in deep learning with pytorch.
***********************************************************************'''
""" 
why deep learning? 
    computer vision is certainly one of the fields that have been most impacted by the
    advant of deep leaning.
    the need to classify or interpret the content of natural images existed.
    very large datasets
    new constructs such as convolutional layers were invented and could be run quickly on GPUs 
    with unprecedented accuracy.
    all of these factors combined with the internet giants' desire to understand pictures taken
    by millions of users with their mobile decices and managed on said giants' platforms.
    so it is the reason why deep learning can do many things.

in this chapter, we will learn three popular pretrained models, one can label an image according
to its content, one can fabricate a new image from a real image, and the last can describe
the content of an image using proper english sentences.
we will learn how to load and run these pretrained models in pytorch.
the first cnn model is AlexNet what is a rather small network, but in our case, 
it is perfect for taking a first peek at neural network that does something and
learning how to run a pretrained version of it on a new thing. we can simple see
the structure of AlexNet. we can show all the models in pytorch used models instance.
we can also use all the models even if model has not added in the torch, we can use
it, because the torch will download the model directly.
these instance variabale can be called like a function. taking as input one or more images
and producing an equal number of scores for each of the 1000 ImageNet classes.

fo course, the preprocessing for the input image is necessary. and pytorch has defined 
the preprocessing function in torchvision module. it is transforms.

we have defined all the content in basic function.
then, we will learn how to generate one picture. just like we need to generate one
world famous painting. we need to find one profession to inspect our work and tell us
exactly what is was that tipped them off that the painting was not legit. with that feedback, 
we can improve our output in clear, directed ways, until our sketchy scholar can no longer
tell our paintings from the real thing.

then, the idea above in the context of deep learning is known as the GAN game(generate adversarial network), 
where two networks, one acting as the painter and the other as the art historian, compete to outsmart each 
there at creating and detectiong forgeries, GAN stands for generative adversarial networkm where generative 
means something is being created. adversarial means the two networks are competing to outsmart the other, 
and well, network is pretty obvious. these networks are one of the most original outcomes of recent deep learning research.
and the more details about the GAN we will learn at last.



we have learned how to generate the pretrained model used torchvision, 
load the pretrained model file used the load_static_state method what we can 
inherite it from the nn.Module class. then, we will learn how to use torch, but 
before that, we must have a solid understanding of how pytorch handles and stores data. as input, as intermediate
representations, and as output. then, it is tensor what the fundamental data structure.
first, you should distinguish the difference between the list in python and the tensor in pytorch.
tensor has the same storage structure to the array in numpy. they are 


learned the storage type, we will learn the numeric type for tensor. the dtype argument to tensor constructors
, the data type specifics the possible values the tensor can hold and the number of bytes per value.
the dtype argument is deliberately similar to the standard numpy argument of the same name. the default data type for 
tensor is 32-bit floating-point. notice it is range from 1.1755e-38 to 3.4082e38 and from -3.4082e38 to -1.1755e-38
notice the precision for the floating-point. it has the positive min precision and max precision, negative
min precision and max precision.
why 32bits? as we will see in future chapters, computations happening in neural networks are typically executed
with 32-bits floating-point precision, higher precision, like 64-bits, will not buy improvements in the accuracy
of a model and will require more memory and computing time. of course, we can use the half-precise 16-bits.
it is suitable for the GPUs not suitable for the cpus. so it is possible to switch to half-precision to decrease
the footprint of a neural network model if needed. of course, it will result in one minor impact on accuracy.

tensor can be used as indexes in other tensors, of course, we should create the integer dtype if we want to use the tensor
as indexes for other tensors. so we should create tensor using torch.tensor([2, 2]), notice it will create the 64-bits
integer for one tensor. it is default dtype for integer. and 32-bits is default for floating-point.
just like torch.tensor([1.0, 1.0]), this dtype is 32-bits floating-point.


then, we will learn the gradient explosion and gradient disappear.
when happend gradient explosion?
first, gradient explosion means the gradient value is so super big that
the loss function will be very big that computer will not show it normally.
then, we should consider to reduce the gradient value or reduce the learning rate.
because they can both influence the loss value. becaue they will influence the weight or
bias first, then we will calculate the loss value based on the weight and bias.
so we can reduce the learning rate or reduce the gradient value to avoid the 
abnormal loss value. but you should known the learning rate can not be too small. or you will waste bigger 
computing resources and lower efficient. and you should known we will update at least two params just like weight and bias
. so one learning rate should suitable for these two params. so you should notice the gradient for weight and bias.
notice the difference between them, if the difference is big, you should normalize the input value. just like you can
scale the input value. input * 0.1, and so on. if you do this, you can use the smaller learning rate to get a better
result.
so for one training, the normalization for the input is necessary. and a suitable learning rate is necessary.
of course, in response to the gradient explosion, we can do the gradient cut.
how to response to the gradient disappear? because we used the chain derivative method. if we used the sigmoid function
what is the nonlinear function, and the biggest gradient value for the sigmoid function is 0.25, and the inited weight
is very small. generally less than 1. so if the layer number is very bigger, the chain derivative value will be close
to zero. so the gradient disappear will happen. so the deeper neural network will result to the gradient disappear.
then, how to handle these problems?
conclusion, initialized the greater weight, the greater learning rate or  deeper nueral network will
result to the gradient explosion. the nonlinear sigmoid function, deeper neural network will result to the
gradient disappear. so the solution is as follow.
1 gradient cut, it is generally used for the gradient explosion.
2 the linear sigmoid funtion, just like relu.  it is generally used for the gradient disappear.
3 the batch normalize, just like scale the input value or (input - mean) / std. it is generally used for  the
gradient disappear and explosion.
4 weight regularization, it can avoid to the gradient explosion.
5 short cut or resnet, it can avoid the gradient disappear.


"""
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image

import torch.nn as nn

class Chapter1:
    def __init__(self) -> None:
        pass
    
    def basic(self):

        # we can print all the model in pytorch and 
        # create one pretrained model used models.
        """
        print(dir(models))
        alexnet = models.AlexNet()
        print(alexnet)
        """

        # we can create the resnet101 used models instance.
        resnet = models.resnet101(pretrained=True)
        # print(resnet)

        # then, we can create the preprocessing instance used transforms in torchvision.
        # notice these normalize paramters what involved mean and std, we have got them when
        # we trained the samples. of course, we should normalize the test dataset used
        # the same mean and std we have got them during training.
        # resize means we will resize the image from the original size to 256*256, 
        # generally, this size is a big size for the original image. so the size of 
        # original image usually has smaller size.
        # reshape the original image from original size to 256*256 first, 
        # then, shear the original image 224*224 based on the center point.
        # you will get one image that shape is 224*224
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225])
        ])

        # we can read one image used pillo tool what is coded used python.
        # notice, this tool will call the image view in your current computer.
        img = Image.open("../../data/images/dog.png")
        # the shape of the image is 640*400
        # print(img.size) # 640*400
        img_pre = preprocess(img)
        print(img_pre.shape) # 3*224*224
        # reshape the image from 3*224*224 to 1*3*224*224. 
        # notice the unsqueeze function has this efficacy.
        batch_t = torch.unsqueeze(img_pre, 0)
        # print(batch_t.shape)
        # resnet.eval()
        # then, you will get one tensor variable that shape is 1*1000.
        # 1000 is because the class numbers of the resnet model is 1000.
        out = resnet(batch_t)
        # print(out.shape) # shape is 1*1000.

        # them, we can find the max score in out variable we have
        # calculated and find the correspond label in class txt.
        # we can read all the classes in class txt file used one list variable.
        with open("../../data/imagenet_classes.txt") as f:
            labels = [line.strip() for line in f.readlines()]
        print(labels[463])
        # torch.max will return the max value and max index in the input tensor.
        # here what we mentions is index not the max value.
        # the second param is the dimension. if the value is 1, it means you want
        # to return the index of maximum value for each row. because the shape of
        # out tensor variable is 1*1000, so we should return the index of the maximum
        # value for each row. notice the return variable index is alse the tensor type.
        # so we can use the index [0] to get the number used python number type. 
        _, index = torch.max(out, 1)
        # then, we can use the index to access the label.
        # notice the difference between softmax and logical regression.
        # the former is dedicated to the many classification problem. and the last
        # is dedicated to the binary classification problems. they can both return
        # the probability for all the classes, and the sum of all the probabilty is
        # equal to 1. you can image them as the sigmoid function.
        # just like z=[z1, z2, ..., zn]
        # softmax(zi) = e^zi / sum(e^zi), for i = 1 to n.
        # notice, e^zi can make all the score to positive.
        # and divide into sum(e^zi) can normalize all the score.
        # the sigmoid function of logical regression, sigmoid(z) = 1 / (1 + e^(-z))
        # notice, z is the prediction value, it is range from 0-∞, we can make the
        # z value to 0,1 used the sigmoid function. notice, the data type of the percentage
        # is also the tensor.
        percentage = torch.nn.functional.softmax(out, dim = 1)[0] * 100
        print(labels[index[0]], percentage[index[0]].item())
        # sort will return the index list what has sort based on the rule you have passed into
        # the sort function. notice indices has the two dimension.
        _, indices = torch.sort(out, descending=True)
        print([(labels[idx], percentage[idx].item()) for idx in indices[0][:5]])



class ResNetGenerator(nn.Module):
    def __init__(self, input_nc = 3, output_nc = 3, ngf = 64, n_blocks = 9) -> None:
        assert(n_blocks >= 0)
        # recall the construct function of the father class.
        # because the ResNetGenerator has inherited the father class nn.Module. 
        # so we should init all the attribution and method by recalling the construct function of the nn.Module first,
        # then, init the attribution and method of the ResNetGenerator.
        super(ResNetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        

class Storage():
    def __init__(self) -> None:
        pass

    def basic(self):
        # calculate the mean of channel for one tensor.
        img_t = torch.randn(3, 5, 5)
        weights = torch.tensor([0.2126, 0.7152, 0.0722])
        batch_t = torch.randn(2, 3, 5, 5)
        # calculate the mean of the tensor based on the bottom third dimension what is the channel dimension.
        image = img_t.mean(-3)
        batch_image = batch_t.mean(-3)

        print(image.shape, batch_image.shape)
        # notice, the difference between unsqueeze and unsqueeze_, the last will change the original variable weights.
        # the former will not change it. and the function of unsqueeze is to add one dimension in corresponding dimension
        # based on the param you have passed. -1 means add the dimension at last. this operation will
        # transform the variable weight from (3, ) to (3, 1, 1)
        unsqueezed_weights = weights.unsqueeze(-1).unsqueeze_(-1)
        # we can multi img_t and unsqueezed_weights directly.
        # (3, 1, 1) * (3, 5, 5) and (2, 3, 5, 5) * (3, 1, 1)
        img_weights = (unsqueezed_weights * img_t)
        batch_weights = (batch_t * unsqueezed_weights)
        img_gray_weighted = img_weights.sum(-3)
        batch_weighted = batch_weights.sum(-3)
        # (5, 5) and (2, 5, 5)
        print(img_gray_weighted.shape, batch_weighted.shape)
        # (2, 3, 5, 5) and (3, 1, 1)
        print(batch_weights.shape, unsqueezed_weights.shape)

        # of course, we can also use einsum to implement this function
        # notice this code as follow. ...mean the batch dimension, chw means the channel dimension, 
        # height and width dimension. c means the channel dimension. so this word means transform the dimension
        # from (batch, channel, height, width) to (batch, height, width), and the detail is to multi each
        # channel and corresponding weight, and sum to get one gray image. it means we will drop the channel dimension.
        batch_gray_weighted_fancy = torch.einsum('...chw, c->...hw', batch_t, weights)
        # so you will find you have got one tensor that shape is (2, 5, 5)
        print(batch_gray_weighted_fancy.shape)

        # of course we can add the name for one tensor when we created it. or you can refine the names
        # for the tensor you have defined succesfully.
        img_named = img_t.refine_names('channels', 'rows', 'columns')
        batch_named = batch_t.refine_names('batch', 'channels', 'rows', 'columns')
        print('img_named: ', img_named.shape, img_named.names)
        print('batch_named: ', batch_named.shape, batch_named.names)
        # of course, we can also make the dimension name of one tensor to aligin as one tensor's dimension name.
        # just like we can define the weight_named what is the named tensor for weights that is aligin as 
        # the tensor's dimension name. of course, you should make sure these two tensor has the same names.
        # if have one name is difference in these two tensor, it will error. and you should notice the align_as
        # function will move the orginal data follow the dimension name. just like
        # a_named(1, 3, 1), ('rows', 'channels', 'columns'), b_named(3, 5, 5), ('channels', 'rows', 'columns')
        # if you code a_named.align_as(b), you will transform a_named as(3, 1, 1), ('channels', 'rows', 'columns'),
        # not (1, 3, 1), ('channels', 'rows', 'columns').
        unsqueezed_weights_only = weights.unsqueeze(0).unsqueeze_(-1)
        weights_named = unsqueezed_weights_only.refine_names('rows', 'channels', 'columns')
        print(weights_named)
        weights_aligin = weights_named.align_as(img_named)
        print(weights_aligin.shape, weights_aligin.names)

        # then, what the function of the names dimension? it can be used like dimension arguments. just like as follow.
        # notice, once you dropped the channels dimension, you will get one gray image.
        # this code is equal to .sum(-3)
        gray_named = (img_named * weights_aligin).sum('channels')
        print(gray_named.shape, gray_named.names)
        # then, what we should do if we want to drop the dimension name for one tensor?
        # we can use rename function.
        unnamed = weights_aligin.rename(None)
        print(unnamed.names)


class DataType():
    def __init__(self) -> None:
        pass



    def basic(self):
        """ 
        integer_tensor = torch.tensor([2, 2])
        floating_tensor = torch.tensor([1.0, 1.0])
        # torch.int64 and torch.float32
        print(integer_tensor.dtype, floating_tensor.dtype)
        # you will generate one bool dtype tensor if you compare the tensor with one number, you can 
        # use integer number or float number. notice, these operation are very convenient for you code.
        bool_tensor = floating_tensor > 0.0
        print(bool_tensor.dtype)
        # of course, you can define the dtype used default, you can also define it by yourself when you 
        # defined one tensor. just like you will get one float64 dtype if you code dtype=double.
        double_tensor = torch.tensor([10, 2], dtype=torch.double)
        print(double_tensor.dtype)
        # you can also define the short what is int16.
        short_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.short)
        print(short_tensor.dtype)
        # notice the long dtype is default integer data type.
        long_tensor = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long)
        print(long_tensor.dtype)
        # we can also use .double() or .short() to cast the current tensor.
        short_double_tensor = short_tensor.double()
        print(short_double_tensor.dtype)
        # we can also use .to(torch.double) to cast the current tensor.


        points_64 = torch.rand(5, dtype = torch.double)
        points_short = points_64.to(torch.short)
        points_32 = torch.rand(5)
        # float64 * int16 = float64, because the integer * float = float. float * double = double.
        print((points_64 * points_short).dtype)
        print((points_64 * points_32).dtype)
        # this concept means when mixing input types in operations, the inputs are converted to the
        # large type automatically, thus, if we want 32-bit computation, we need to make sure all our inputs
        # are at most 32-bit. so it means the largest data type in your input data is 32-bit, you can not
        # input one 64-bit data type. or your input will be 64-bit when you have mixing in operations.
        # you can use transpose function for one tensor, of course, you should input the dimension what you want to
        # transpose.
        a = torch.ones(2, 3)
        a_t = a.transpose(0, 1)
        print(a.shape, a_t.shape) 
        """
        
        # then, we will consider the storage concept in torch. multiple tensor can share one storage in torch if
        # they are created from the same data or if they are dirivated from the same source tensor.
        # tensor.storage() method returns the one-dimensional underlying data storage of a given tensor.
        # the data storage is a contiguous array that holds the actual elements of the tensor. 
        # keep in mind that changing elements of the storage directly affects all the tensors sharing the same
        # storage, therefore you should known when you should copy one new tensor to avoid to change the tensor.
        
        
        
        """         
        tensor_a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        storage_a = tensor_a.storage()
        print(storage_a, storage_a.dtype)
        # then, we can try to define one sub tensor based on the tensor_a
        tensor_b = tensor_a[:3]
        tensor_b[0] = 1e-3
        # you will find tensor_a has changed the similar like tensor b. so tensor b and tensor a shared one storage.
        # so you should notice this concept. storage. and of course you can change the storage value. you will find
        # all shared tensor based on the storage will be changed if you changed the storage value.
        # the storage is 1 dimension.
        storage_a[2] = 3e-10
        print(tensor_b, tensor_a)
        # zero_, *_, which indicates that the method operates inplace by modifying the input instead of creating
        # a new output tensor and returning it.

        points = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        # notice, if you do not want the subtensor to share the same storage with the points tensor. you
        # should clone one new tensor for the subtensor second_points. or your changes for the subtensor
        # will also influence the points tensor.
        second_points = points[1].clone()
        second_points[0] = 10.0
        print(second_points, points)
        # of course, you can use the tensor.t() function to transpose one tensor.
        points_t = points.t()
        print(points_t)
        a = torch.tensor([11, 12])
        first_points = points[0]
        # then, how to verify that the two tensors share the same storage?
        # you can compare the id of these two tensor's storage.
        # just like the case above, points and second_points.
        print(id(points.storage()))
        print(id(points_t.storage()))

        # notice the tensor.stride() attribution, it can transform one storage to one tensor.
        # we all known one storage is 1 dimension, then we can give one tensor based on one storage and stride.
        # just like the storage(1, 2, 3, 4, 5, 6), if the stride is (3, 1), then the tensor will be
        # [[1, 2, 3],
        # [4, 5, 6]]
        # the first dimension of stride means you should skip 3 element if you want to go to next row.
        # the second dimension of stride means you just need to skip 1 element if you want to go to next column. 
        stride_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        print(stride_tensor.stride())
        # the transpose function in torch is not limited to matrices. we can transpose a multidimensional array
        # by specifying the two dimensions along with transposing.
        some_t = torch.ones(3, 4, 5)
        # you can specific the two dimension what you want to transpose. of course, the stride will be also
        # transposed with the tensor itself.
        transpose_t = some_t.transpose(0, 2)
        stride_some = some_t.stride()
        stride_transpose = transpose_t.stride()
        print(some_t.shape, transpose_t.shape)
        print(stride_some, stride_transpose)
        print(some_t.is_contiguous())
        print(transpose_t.is_contiguous())

        # some tensor operations in pytorch only work on contiguous tensors, in our case, some_t is 
        # contiguous, while it transpose is not. we can obtain one new contiguous tensor from a non-contiguous
        # one using the contiguous method. the content of the tensor will be same. but the stride will be changed.
        # notice the stride is based on the storage. just like the case as follow:
        points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
        points_t = points.t()
        print(id(points.storage()), id(points_t.storage()))
        print(points.stride(), points_t.stride())
        # false
        print(points_t.is_contiguous())
        points_contiguous = points_t.contiguous()
        # notice, they have the same array after contiguous. 
        # but the stride will be not same.
        print(points_contiguous, points_t)
        # the stride has changed from (1, 2) to (3, 1)
        print(points_contiguous.stride(), points_t.stride())
        # you can find the reason is that the storage has changed.
        print(points_contiguous.storage()) 
        """


        # then, how to move one tensor to GPU
        # we have learned all basic operation for tensor in CPU, then, we will consider it in GPU
        # pytorch tensors also can be stored on a different kind of processor: a graphics processing unit(GPU)
        # every pytorch tensor can be transferred to the GPU in order to perform massively parallel, fast computations.
        # all operations that will be performed on the tensor will be carried out using GPU-specific routines that
        # come with pytorch.
        # we can store the tensor on GPU by managing a tensor's device attribution.
        """
        points_gpu = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]], device='cuda')
        # of course, we can move the tensor in cpu to GPU
        points_cpu = torch.tensor([1.0, 2.0, 3.0])
        cpu_gpu = points_cpu.to(device='cuda')
        result = points_gpu * 2
        # the result will store on GPU, and you should specific the device is equal to cpu if you want to make it back to cpu.
        # notice, you can specific which GPU you want to use if you have multi GPU in your computer, just like cuda:0
        # this is the first GPU in your computer, you can also specifc cuda:1, of course, we can also
        # use the shorthand method to move the tensor to GPU or come back to cpu
        gpu_cpu = points_gpu.cpu()
        aaa = gpu_cpu.cuda(1) 
        """
        # numpy interoperability.
        # which will return a numpy multidimensional array of the right size, shape and numerical
        # type, and interestingly, the returned array shares the same underlying buffer with the tensor
        # storage, this means modifying the numpy array will lead to a change in the originating tensor. if the tensor
        # is allocated on the GPU, pytorch will make a copy of the content of the tensor into a numpy array allocated
        # on CPU. of course, we can obtain a pytorch tensor from a numpy array.
        """
        points = torch.ones(3, 4)
        points_numpy = points.numpy()
        print(points)
        print(type(points_numpy))

        points_ = torch.from_numpy(points_numpy)
        print(points_)
        # the points_numpy will be changed if you changed the points. because they used the same buffer-sharing
        # strategy we just described.
        points[1, 2] = 10000
        print(points, points_numpy)

        # this stored file is binary type.
        # you can save the points to disk used these two method
        # torch.save(points, '../../data/files/points.txt')
        # with open('../../data/files/pointss.txt', 'wb') as f:
        #     torch.save(points, f)
        # of course, you can load the disk file into memory.
        # torch.load('../../data/files/points.txt')
        # with open('../../data/files/points.txt', 'rb') as f:
        #   points = torch.load(f)
        # rb means read binary. wb means write binary.
        """


        import imageio
        img_arr = imageio.imread('../../data/images/dog.png')
        img = torch.from_numpy(img_arr)
        # set the channel dimension to the first dimension.
        # note that this opearation does not make a copy of the tensor data. instead, out used the same
        # underlying storage as img and only plays with the size and stride information at the tensor level.
        # this is convenient because the operation is very cheap,  but just as head-up, changing a pixel
        # in img will lead to a change in out.
        # so far, we have described a single image. following the same strategy we have used for
        # earlier data types, to create a dataset of multiple images to use as an input for our neural
        # networks, we store the images in a batch along the first dimension to obtain an N*C*H*W tensor.
        out = img.permute(2, 0, 1)
        print(out.shape)

        batch_size = 3
        batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)
        # then, we have preallocated one variable batch, we can now load all png images from
        # an input directory and store them in the tensor.
        import os
        data_dir = '../../data/test/'
        filenames = [name for name in os.listdir(data_dir)]
        for i, filename in enumerate(filenames):
            image_arr = imageio.imread(os.path.join(data_dir, filename))
            image_t = torch.from_numpy(image_arr)
            image_t = image_t.permute(2, 0, 1)
            image_t = image_t[:3]
            batch[i] = image_t
        # util here, we have preloaded our images in the tensor. but note that the data type. we mentioned 
        # earlier that neural networks usually work with floating-point tensors as their input. 
        # neural networks exhibit best training performance when the input data ranges roughly from 0 to 1, 
        # or from -1 to 1. so a typical thing we will want to do is cast a tensor to floating-point and normalize
        # the values of the pixel. the former is easy, the last is trickier. as it depends on what
        # range of the input we decide should lie between 0 and 1 or -1 and 1. one possibility is to just
        # divide the values of the pixel by 255, the range will changed to (0, 1)
        batch = batch.float()
        # range from 0 to 1
        # batch /= 255.0
        # print(batch)

        # of course, another possibility is to compute the mean and standard deviation of the input data
        # and scale it so that the output has zero mean and unit standard deviation across each channel:
        # of course, we should calculate them for each channels. so it means we should calculate the average
        # value of all the samples for each channel. R, G and B channel.
        # n_channels = batch.shape[1]
        # for c in range(n_channels):
        #     mean = torch.mean(batch[:, c])
        #     std = torch.std(batch[:, c])
        #     batch[:, c] = (batch[:, c] - mean) / std
        # print(batch)

        # we can perform several other operations on inputs, such as geometric transformations like rotations
        # scaling, and cropping.


class FirstModel():
    def __init__(self) -> None:
        pass

    # t_u is the horizontal axis, t_c is the vertical axis.
    def model(self, t_u, w, b):
        return w * t_u + b
    
    # t_p is the predict value and t_c is the actual value.
    def loss(self, t_p, t_c):
        squared_diffs = (t_p - t_c)**2
        return squared_diffs.mean()

    # the rate for w is equal to [loss(model(t_u, w+delta, b), t_c) - loss(model(t_u, w-delta, b), t_c)] / (2*delta)
    # but these two method to calculate the rate for w and b parameters is inaccurate, because we have the super
    # param delta, we can not very good to define its size. so we can calculate the derivative of the loss function
    # for w and b.
    def rate_w(self, t_c, t_u, w, b):
        delta = 0.1
        loss_rate_of_change_w = (self.loss(self.model(t_u, w + delta, b), t_c) - 
                                    self.loss(self.model(t_u, w - delta, b), t_c)) / (2.0 * delta)
        return loss_rate_of_change_w
    

    def rate_b(self, t_c, t_u, w, b):
        delta = 0.1
        loss_rate_of_change_w = (self.loss(self.model(t_u, w, b + delta), t_c) - 
                                    self.loss(self.model(t_u, w, b - delta), t_c)) / (2.0 * delta)
        return loss_rate_of_change_w

    # dloss / dw = (dloss / dt_p) * (dt_p / dw)
    # dx2/dx = 2x
    def d_loss(self, t_p, t_c):
        dsquare_diffs = 2 * (t_p - t_c) / t_p.size(0)
        return dsquare_diffs

    # t_p = w * t_u + b
    # dt_p / dw = t_u
    # dt_p / db = 1
    def dt_p_w(self, t_u, w, b):
        return t_u

    def dt_p_b(self, t_u, w, b):
        return 1.0
        
    # return one tensor(2, ), the first is the gradient dloss / dw, and the second is the gradient dloss / db
    # each pixel will corresponding to one gradient, we will return the sum value as the gradient value
    # for dw or db. notice, the dimension of dw and db is equal to w and b.
    def grad_function(self, t_u, t_c, t_p, w, b):
        dloss_dt_p = self.d_loss(t_p, t_c)
        dloss_dw = dloss_dt_p * self.dt_p_w(t_u, w, b)
        dloss_db = dloss_dt_p * self.dt_p_b(t_u, w, b)
        return torch.stack([dloss_dw.sum(), dloss_db.sum()])
        
    # then, we can define the train function. 
    # this params stored the w and b what are both the optimized param.
    # notice t_u is the horizontal axis, and t_c is the vertical axis.
    # notice the learning_rate, it should be smaller, or the descent step will be large so that
    # the loss function will be inf. but you will find another problem that the loss value will down very
    # slowly if the learning_rate is small enough. so we should notice it and of course, we have another
    # problem in the update term. we can find the gradient for the weight is 50 times greater than
    # the gradient for the bias tensor([4517.2964,   82.6000]) in the epoch 1. this means the weight and 
    # bias live in differently scaled spaces. if this is the case, a learning rate that's large enough to meaningfully
    # update one will be so large as to be unstable for the other. there is a simple way to keep things in check.
    # changing the inputs so that gradients are not quite so different. we can make sure the range of the input
    # does not get too far from the range of -1.0 to 1.0. roughly speaking, in our case, we can achieve something
    # close enough to that by simply multiplying t_u by 0.1.
    def training_loop(self, n_epochs, learning_rate, params, t_u, t_c):
        for epoch in range(1, n_epochs + 1):
            w, b = params
            t_p = self.model(t_u, w, b)
            loss = self.loss(t_p, t_c)
            grad = self.grad_function(t_u, t_c, t_p, w, b)
            params = params - learning_rate * grad
            print('Epoch %d, Loss %f' %(epoch, float(loss)))
            print(params)
            print(grad)
        return params



    # then, we will use the autograd in pytorch. we just need to define the params used torch.
    def training_loop_autograd(self, n_epochs, learning_rate, params, t_u, t_c):
        for epoch in range(1, n_epochs + 1):
            # reset the params.grad
            if params.grad is not None:
                params.grad.zero_()
            t_p = self.model(t_u, *params)
            loss = self.loss(t_p, t_c)
            loss.backward()
            # temporary set all the require_grad to False.
            # we just need to update the params, not need to calculate its gradient.
            with torch.no_grad():
                params -= learning_rate * params.grad
            if epoch % 500 == 0:
                print('Epoch %d, Loss %f' %(epoch, float(loss)))
        return params
    

    # of course, we can define the optimizer based on the pytorch.
    # notice, you should define the learning rate and the gradient descent method for the optimizer.
    def training_loop_optimizer(self, n_epochs, optimizer, params, t_u, t_c):
        for epoch in range(1, n_epochs + 1):
            t_p = self.model(t_u, *params)
            loss = self.loss(t_p, t_c)
            # notice reset the grad used zero before each update.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 500 == 0:
                print('Epoch %d, Loss %f' %(epoch, float(loss)))
        return params








