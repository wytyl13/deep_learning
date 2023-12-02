/**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-08-07 10:58:06
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-08-07 10:58:06
 * @Description: 
 * 
 * 
 * 
 * 解决github无法连接错误 OpenSSL SSL_connect: Connection was reset in connection to github.com:443
 * this error generally happend when you used ssl to connect github. and you 
 * used the proxy. so you should define the proxy for git config.
 * git config --global http.proxy 127.0.0.1:7890
 * git config --global https.proxy 127.0.0.1:7890
 * 7890 is the port of your proxy.
 * and you should git add *, git commit -m "any string", git push -u origin main.
 * or you will get the same error when you git push your program.
 * 
 * 
 * why batch normalize ？
 * 目的是加速模型训练过程并提高模型的收敛性和稳定性。
 * 主要是通过解决神经网络训练过程中的内部协变量偏移问题来达到这种目的。
 * 什么是内部协变量偏移：神经网络的每一层在训练过程中会受到前一层输入数据分布的影响，这样会导致每一层的输入分布发生变化。
 * 这个称之为内部协变量偏移，这种问题会使得网络的学习变得困难。需要更小的学习率和更复杂的调参。而batch normalize
 * 通过在每个小批量的数据上进行归一化，使得每一层输入的分布稳定。有助于加速训练过程。
 * 优点：
 *      加速收敛速度。
 *      允许使用较大的学习率。
 *      减少过拟合。在一定程度上可以起到正则化的作用。
 *      增强模型的泛化能力。
 *      降低初始化对结果的影响。
 * 当然，batch normalize也增加了计算的复杂性。并且在小批量大小的情况下不适用。、
 * 为什么batchnormalize可以减少过拟合？
 * 因为batch normalize会引入噪声。类似于dropout的效果
 * 限制了权重的过度增长。如果模型中的权重大小没有极端情况，那么模型将不会存在过拟合现象。
 * 减少梯度消失问题。因此可以减少因为梯度消失引起的训练困难和过拟合。当然过度使用batchnormalize也可能导致欠拟合。
 * sigmoid通常用于二分类问题的输出层，而sofmax和交叉熵损失函数结合用于多分类问题。
 * sigmoid是线性激活函数。它的取值范围是0到1.梯度值为0到0.25，当输入接近0的时候哦梯度最大，反之无限接近0.所以
 * sigmoid函数很容易出现梯度消失问题。
 * 
 * 
 * 
 * 还有一个问题就是为什么要归一化
 *      因为我们在每一层至少要学习两种参数，权重和偏置。而我们只能定义一个学习率。
 *      我们要选择适合两种参数的学习率。学习率既不能过大也不能过小，过大的学习率会导致
 * 梯度爆炸，因为如果在凸优化问题中学习率过大，会导致损失函数在优化过程中跳过最优解，如果此时
 * 梯度值特别大，那么参数的更新会导致损失值剧烈变化，甚至越来越大。这就是梯度爆炸的问题。为什么？
 * 因为学习率首先会通过较大的学习步伐来影响到下一步的梯度值的大小，因此学习率过大会直接导致梯度
 * 爆炸的问题产生，而如果梯度爆炸的问题产生，对应的损失函数也会无限放大。
 * 而学习率过小则会产生梯度消失的问题。因为前后两步的变化率很小。同时学习率较小可能会陷入局部最优解
 * 。而且收敛速率会下降。因此学习率不能过大也不能过小。
 *      那么对应两种类型的参数，找到一个适合两个参数的学习率是很困难的。因为偏置我们一般是随机
 * 初始化，它的分布一般是根据我们初始化的方法定的，一般会较小。而输入则由输入数据来定。
 * 如果两者的分布和数值出现很大的差距，会造成两者的梯度差距很大。比如300-250，它的梯度是
 * 50/300=1/6, 而同样的3-2.5， 它的梯度是0.5/3=1/6.虽然相同的梯度，但是他们之间的变化的
 * 数值是差距很大的，如果同样的数值变化，则会造成他们之间很大的梯度差距，此时会产生梯度爆炸的问题。
 * 因为同样的学习率，在偏置参数来说不会产生梯度爆炸，而在权重参数来说就会产生梯度爆炸，所以此时
 * 我们可以对输入数据做归一化处理，使得它的分布更加接近偏置参数。那么同一个学习率会适合两种
 * 类型的参数，梯度爆炸问题也会解决。
 * 
 * 
 * 因此，较大的初始化权重，较大的学习率和较深的网络都会导致梯度爆炸
 * 而线性激活函数，更深的网络和较小的学习率则会导致梯度消失。
 * 解决办法如下：
 *      梯度裁剪。可以被应用于梯度爆炸
 *      使用非线性机会函数，比如relu可以解决梯度消失的问题。
 *      batchnormalize可以解决梯度爆炸和梯度消失问题。
 *      权重正则化，可以防止梯度爆炸问题。正则化的效果在于当梯度更新模型的参数时，对于较大的
 *      权重，正则项会施加更强的抑制作用。从而限制权重的增长幅度。从而预防梯度爆炸。
 *      renet或者shortcut，可以防止梯度消失问题。

 * 那么正则化如何在损失函数中作用的呢？
 * 首先正则化可以降低过拟合，同时可以减少梯度爆炸情况。
 * 模型过拟合的一个重要原因是模型太过复杂，可学习的参数很多。那么此时我们可以通过减少模型的参数
 * 来降低模型的复杂度来减少模型的过拟合。这里强调一点，模型的参数越小其实越好，因为越大的参数
 * 会造成越大的梯度。所以参数越小模型就越简单，但是还有一个指标来衡量模型的好坏，就是模型各个
 * 参数的分布情况，不能出现极端值。也即不能出现某个权重过大的情况。分布均匀才是最好的。
 * 首先笼统的讲，正则化可以简化模型，也就意味着正则化可以较少模型的复杂度和模型的容量。
 * 他可以减少模型的参数通过增加一些条件使得参数为0或者接近0.但是这并不是终极办法，就像我们上面提到的
 * 使得参数为0没有使得参数分布更加均匀的效果好。如下
 * w1, [1, 0, 0, 0]
 * w2, [0.25, 0.25, 0.25, 0.25]
 * x = [1, 1, 1, 1]
 * 上面两组权重当然是第二个更加好。当然学习的参数值越小越好。
 * 如w1.T@x = 1, and w2.T@x = 1, 结果相同，但是w2的效果更好。然而为什么正则化可以减少过拟合呢？
 * 因为L1加入了一些条件使得参数值为0，而L2则加入了一些条件使得参数值接近0.
 * 
 * 
 * 两个重要的损失函数：
 *      softmax：L = -log(e^s/Σe)
 *          first, we should transform the score from value to probability.
 *          we should use cross entropy, e^score/Σe^score, this is probability.
 *          then, we use -log(probability of actural label)
 *          cat, dog, person(3, 5, 1), the actual label is cat. then the
 *          softmax loss value is eqaul to -log(e^3/(e^3+e^5+e^1)). we can find, if
 *          the score value is bigger, the softmax loss value will be smaller.
 *          because the probability value is range from 0 to 1, then if the probability
 *          is close to 1, then the loss value will be close to zero. of course, because
 *          the softmax need to calculate each probability for all classes for one sample.
 *          so we should transform the label from one to one-hot type. then we can calculate
 *          the loss.
 *      hinge: max(0, sj - syi + 1), sj is the value that each predict false, syi is
 *      the value the predict true. if the predict value is [3, 2, 1] and correspond to
 *      [cat, dog, people], if the true label is cat, it mean we have predict true.
 *      then the hinge loss is equal to max(0, 2-3+1)+max(0, 1-3+1) = 0
 *      if we have predict false, if the true label is dog. then the hinge loss is,
 *      max(0, 3-2+1)+max(0, 1-2+1) = 2+0=2.so if you have predict false, you will get
 *      the bigger loss.
***********************************************************************/




 * /**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-08-06 08:10:47
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-08-06 08:10:47
 * @Description: consider the program will train used google colab, yolov5
 * so this error will happen when you have trained your program in your
 * computer and you update your program to colab directly, because the cache file in
 * images is your computer's cache.
 * error:AssertionError: Image Not Found data\images\train\5.jpg
 * we can find the dataset file should find image from ..\data\images\train\5.jpg
 * but it has not ../, so the error happend.
 * 
 * we can use the autoanchor tool in utils in yolov5 program to calculate your anchors
 * based on your train data.
 * 
 * of course, we can do some image enhancement based on yolov5.
 * we have defined one script to enhancement the image and generate large
 * numbers image based on the original image.
 * 
 * if you want to update the local program to the cloud. just like google colab
 * you should delete the cache file. or you will get read path error.
 * 
 * if you want to run your program in cloud. you should modify your loss.py
 * in utils dictory. modify these two code as follow.
 * anchors, shape = self.anchors[i], p[i].shape , change 178 rows.
 * indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  
 * change this code.




 * training process
 * update the program. you should change the class_name, num_class and anchors.
 * of course, you can get the achors based on your train data. utils/autoanchor.py
 *      data/voc.yaml, models/yolov5m.yaml
 * step1: make labels. to generate xml files.
 *      store the original image and correspond xml file into the dictory.
 * step2: enhancement and augmented. data_enhancement.py
 * step3: xml to txt and divided into train, val and test. xml_txt.py
 * training: train.py
 * validation. val.py
 * detect: detect.py
 * 
 * 
 * 
 * MLP: multilayer perception or full connection forward neural network.
 * 
 * 
 * if you have trained the model in GPU, then, you want to go on training in cpu.
 * you can change the parameter in torch.load, add 
 * ckpt = torch.load(weights, map_location=torch.device('cpu'))
 * run_id = torch.load(weights, map_location=torch.device('cpu'))
 * then, you can put the trained weight and model in cpu, then you can train used cpu. or
 * you will get the error:
 *      cuda is not avalible.
 * you can use the code to go on training. based on the last trained weight file as follow.
 * !python train.py --resume runs/train/exp6/weights/last.pt
 * 
 * 我们已经学习了yolov3模型，下面我们来了解下yolov5。
 * yolov5较yolov3更加高效，简洁。
 * 自上而下和自下而上的特征融合。分为主干网络和头部网络，主干网络进行下采样提取特征。
 * 然后从高维特征自上而下的和主干网络进行特征融合，然后又进行下采样，自下而上的和第一个头部
 * 进行特征融合，最后输出三个预测。
 * 
 * 
 * 下采样
 * 
 * input（640*640*3） --> downsample --> upsample and concact with the former backbone --> 
 * downsample and concat with the former upsample --> detect based on three scale.
 * 13, 26 and 52.
 * 
 * 这样的双结合的样式也可以称之为双塔结构。
 * 主干结构主要由普通卷积、C3和SPPF构成
 * C3部分主要由卷积和残差组成，bottleneck，并且当输入和输出通道数不一致的时候bottleneck存在差异
 * SPPF主要是空间金字塔快速池化，它的目的就是高效率。



 * 如果你收到了这个错误消息
 * from models.common import *
 * ModuleNotFoundError: No module named 'models'
 * 这个是因为你在当前终端执行yolo.py文件的时候，models目录是当前终端的上一级目录。所以
 * 肯定找不到这个路径，然后你可以在和models平级的终端目录下运行yolo.py文件。
 * 包是文件夹，包里面的文件是模块。模块中有类函数和变量都可以导入。
 * 相对路径只能使用from导入不能使用import. 并且不能再入口文件中使用相对路径。
 * 这里需要区分入口文件和模块文件的区别，不是是否有入口函数的区别，如果一个入口文件被另一个
 * 入口文件import，那么它就是模块文件。只有我们在执行某一个入口文件的时候，他才是入口文件。
 * 注意入口文件不能使用相对路径导入包。
 * 一个包就是一个文件夹，该文件夹下必须存在init文件，可以为空，该文件夹可以标识当前文件夹是一个包。
 * 
 * 
 * tph-yolov5 基于transformer prediction head改进的小目标检测模型
 * yolov5l-xs-tr-cbam-spp-bifpn 这个是包含了tph和cbam改进
 * yolov5l-xs-tph 这个仅有tph改进
 * 注意如果想要使用这个模型训练自己的数据，要在yolov5l.pt预训练权重的基础上进行训练，因为
 * 该模型师基于yolov5l深度的模型。对应的yolov5l-xs-tr-cbam-spp-bifpn.yaml文件中深度和
 * yolov5l.yaml的深度相同。
 * 注意该模型也可以添加预训练选择框。注意该模型有4个检测头，因为添加的那个160检测头专门用于小物体检测。
 * 
 * 
 * 
 * RuntimeError: Unable to find a valid cuDNN algorithm to run convolution
 * 批次太大，更换小批次。
 * 
 * 
 * pt2onnx
 * python export.py --weights runs/train/exp3/weights/best.pt --include onnx --opset 12 --dynamic
 * 然后将生成的onnx简化，需要下载工具包
 * pip install onnx coremltools onnx-simplifier
 * 使用工具包简化onnx
 * python -m onnxsim best.onnx best-sim.onnx
 * 测试onnx是否可用
 * python detect.py --weights best-sim.onnx --source path/to/img.jpg
 * 
 * 
 * gdb的时候如何添加命令行参数？
 * 可以先使用gdb运行exe文件，然后使用set args param1 param2
 * 然后运行run执行
 * 
 * pt转onnx
 * 首先需要将该文件放在根目录下，如和export文件放在一块。当然最简答的是使用export文件进行转换
 * 注意参数的设定，一定要和自己训练时候的参数一致。
 * 但是需要注意的是转换完一定要在python上测试一下
 * 转换
 * python export.py --weights runs/train/exp3/weights/last.pt --data data/VOC.yaml --include onnx --dynamic
 * 
 * 测试
 * python --weights detect.py runs/train/exp3/weights/last.onnx --dnn
 * 如果测试不通过，那么就是转换参数的设置问题。
 * 
 * 
 * Traceback (most recent call last):
  File "detect_exe.py", line 24, in <module>
  File "ntpath.py", line 703, in relpath
ValueError: path is on mount 'C:', start on mount 'D:'
[26172] Failed to execute script 'detect_exe' due to unhandled exception!

* 这个错误发生在使用pyinstaller打包项目的时候，解决办法就是不要用relpath这个函数
* 同样的错误还会发生，解决办法都是禁用这个函数即可。
* 如果你不想使用一些数据处理方法在yolov5中，你可找到hyp文件夹，更改其下的hyp.scratch.yaml
* 文件，比如如果不想使用mosaic， 那么设置它的值为0.
* ls | wc -w 查看当前文件夹下的文件和文件夹数量。


* 那么现在我们有一个问题，就是我们在使用labelimg对小物体进行标记的时候很容易出现
* 重复标记的情况。解决办法就只能从xml中找到重复的坐标然后删除。我们在unet中是将labelme
* 生成的json文件转换为灰度图像。这里我们需要再将xml文件转换为yolo可以识别的txt文件前，先
* 去除重复标记的坐标。


git rm -r --cached .
git add .
git commit -m "Update .gitignore"
git push -u origin main
***********************************************************************/