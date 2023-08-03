'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-07-31 11:04:56
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-07-31 11:04:56
 * @Description: 
 the size after convolusion is equavalent to (F - W + 2P / S) + 1

 why need to normalization the input?
 just like the sigmoid function, its value is range from 0 to 1.
 and sigmoid(0) = 1/(1+e^0) = 0.5. the range from -3 to 3 is the result value
 for sigmoid function the fastest part of change. but out of the range, 
 the rate is almost zero. so out of the range from -3 to 3, the derivative of the
 sigmoid function will be close to zero. the gradient disappear will happen.
 then, how to handle this problem? we can normalize the input value. we can normalize the
 input value from minus infinity~ infinite to a smaller range, just like -3~3. then, we 
 will avoid the gradient disappear. (input - mean) / std. so this method is dedicated to
 handle the linear sigmoid function, the nonlinear sigmoid function need not. just like relu.
 its derivative is always big. another reason we used batchnormalization after convolution layer
 is it can reach the efficient like regularization. it can instead of the drop out
 in full connnection layer. notice, it means batch normalization can avoid the overfitting.
 and it can reduce the computer amount and improve the efficient for the model running.
 so conv+batchnormal+maxpool is necessary for our training.
***********************************************************************'''


import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms



class Start(nn.Module):
    def __init__(self) -> None:
        super(Start, self).__init__()
        # define the first convolusion layer used the Conv2d function in torch.nn what encapsulates the conv2d
        # function in torch.nn.functional moudle.
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 3, stride = 1, padding = 0)
        # notice, if ceil_mode is true, it means we will record the max value for the scanned area if the area
        # is less than the kernel size. notice the stride for the pool layer is fixed, it is the pool kernel size.
        # so generally the size after maxpooling is equavalent to inputsize / stride. so if the stride is 2, then,
        # the size after pooling is inputsize / 2. and generally size that after convolution is equavalent to the
        # input size, because we used the padding.
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=False)
        self.relu1 = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()
        # transform from 196608 to 10.
        self.linear1 = nn.Linear(196608, 10)

    def load_data(self):
        dataset = torchvision.datasets.CIFAR10("../computerVision/data", train=False, 
                                               transform=torchvision.transforms.ToTensor(),
                                               download=True)
        dataloader = DataLoader(dataset, batch_size=64)
        return dataloader


    def forward(self, x):
        # x = self.conv1(x)
        # output = self.maxpool1(x)
        # output = self.relu1(x)
        # output = self.sigmoid1(x)
        output = self.linear1(x)
        return output


# then, we will define the simplest neural network what have three convolusion and three maxpool. and one
# hidden layer and one output layer.
# we should give the neural network first.
# inputs: 3*32*32 -> 
# conv1: 32*32*32(5*5 kernel) -> 
# maxpool1: 32*16*16(2*2 kernel) ->
# conv2: 32*16*16(5*5 kernel) ->
# maxpool2: 32*8*8(2*2 kernel) ->
# conv3: 64*8*8(5*5 kernel) ->
# maxpool3: 64*4*4(2*2 kernel) ->
# flatten input to the full connection layer. 1024 ->
# scale to 64 hidden neurual. -> 
# output layer, 10 class.
# notice we should calculate the stride and padding based on the convolusion result and input size.
# because we should give the kernel size, padding and stride params into the convolusion function for the nn.Conv2d.
class Cifar10(nn.Module):
    def __init__(self) -> None:
        super(Cifar10, self).__init__()
        # self.conv1 = nn.Conv2d(3, 32, 5, 1, 2)
        # self.maxpool1 = nn.MaxPool2d(2)
        # self.conv2 = nn.Conv2d(32, 32, 5, 1, 2)
        # self.maxpool2 = nn.MaxPool2d(2)
        # self.conv3 = nn.Conv2d(32, 64, 5, 1, 2)
        # self.maxpool3 = nn.MaxPool2d(2)
        # self.flatten = nn.Flatten()
        # self.linear1 = nn.Linear(1024, 64)
        # self.linear2 = nn.Linear(64, 10)

        # or you can use sequential function
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.model1(x)
        return x
    def load_data(self):
        dataset = torchvision.datasets.CIFAR10("../computerVision/data", train=False, 
                                               transform=torchvision.transforms.ToTensor(),
                                               download=True)
        dataloader = DataLoader(dataset, batch_size=64)
        return dataloader

if __name__ == "__main__":
    # start = Start()
    # output = start.forward(1)
    # print(output)

    # test the conv2d
    """     
    input = torch.tensor([
                [1, 2, 0, 3, 1], 
                [0, 1, 2, 3, 1], 
                [1, 2, 1, 0, 0],
                [5, 2, 3, 1, 1],
                [2, 1, 0, 1, 1]
            ])
    kernel = torch.tensor([
                [1, 2, 1],
                [0, 1, 0],
                [2, 1, 0]
            ])
    print(input.shape, kernel.shape)
    input = input.reshape((1, 1, 5, 5))
    kernel = kernel.reshape((1, 1, 3, 3))
    print(input.shape, kernel.shape)
    output = F.conv2d(input, kernel, stride=1, padding=1)
    print(output) 
    """
    # start = Start()
    # writer = SummaryWriter("logs")
    # dataloader = start.load_data()
    # step = 0
    # for data in dataloader:
    #     image, target = data
    #     output = start(image)
    #     print(image.shape)
    #     print(output.shape)
    #     output = torch.reshape(output, (-1, 3, 30, 30))
    #     # notice, add_image can just show one image, and add_images can show many images.
    #     writer.add_images("input", image, step)
    #     writer.add_images("output", output, step)
    #     step = step + 1
    # writer.close()

    # notice, the tensor type should be float, or the error will happen.
    # ceil_mode will influence the max pool result.
    """ 
    input = torch.tensor([
            [1, 2, 0, 3, 1], 
            [0, 1, 2, 3, 1], 
            [1, 2, 1, 0, 0],
            [5, 2, 3, 1, 1],
            [2, 1, 0, 1, 1]
        ], dtype=torch.float32)
    input = input.reshape((-1, 1, 5, 5))
    start = Start()
    output = start.forward(input)
    print(input.shape)
    print(output.shape)
    # if the ceil_mode is True, the result is 2, 3, 5, 1.
    # if the ceil_mode is False, the result is 2.
    print(output) 
    """

    # input = torch.tensor([
    #         [1, -0.5],
    #         [-1, 3]
    #     ])
    # input = torch.reshape(input, (-1, 1, 2, 2))
    # start = Start()
    # ouutput = start.forward(input)
    # print(ouutput)

    """     
    start = Start()
    dataloader = start.load_data()
    writer = SummaryWriter("logs")
    step = 0
    for data in dataloader:
        images, targets = data
        writer.add_images("sigmoid_input", images, step)
        output = start(images)
        writer.add_images("sigmoid_output", output, step)
        step = step + 1
    writer.close() 
    """
    """     
    start = Start()
    dataloader = start.load_data()
    for data in dataloader:
        images, targets = data
        output = torch.flatten(images)
        print(output.shape)
        output = start(output)
        print(output.shape) 
    """

    # test the cifar10 neural network.
    # of course, you can also imshow your neural network structure used tensorboard
    # cifar10 = Cifar10()
    # input = torch.ones((64, 3, 32, 32))
    # output = cifar10(input)
    # writer = SummaryWriter("logs")
    # writer.add_graph(cifar10, input)
    # writer.close()
    # we can use crossentropyloss function to calculate the loss value.
    # crossEntropyloss(x, class) = -x[class] + log(Σexp(x[i]))
    # just like the class has three, 0, 1, 2. then, the predict of course has three probability.
    # the predict result is [0.2, 0.8, 0.1] and the label is 1. x[1] = 0.8
    # the crossEntropyloss value is equavalent to -0.8 + log(exp(0.2) + exp(0.8) + exp(0.1))
    # you can find, if the predict is true, the x[1] is the biggest probabilty, so the crossEntropyloss value
    # will be close to zero.
    """     
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cifar10.parameters(), lr = 1e-2)
    dataloader = cifar10.load_data()
    for epoch in range(20):
        running_loss = 0.0
        for data in dataloader:
            images, targets = data
            output = cifar10(images)
            result_loss = loss(output, targets)
            # you should reset all the parameters' gradient as zero in your model.
            optimizer.zero_grad()
            result_loss.backward()
            optimizer.step()
            running_loss = running_loss + result_loss
        print(running_loss) 
        """



    # of course, we can define the model by ourselves, we can also define the model based on
    # the classic model that pytorch has defined successfully. just like alexnet, vgg and so on.
    # then, we will try to define one new model by modifying some information from the existing model
    # in pytorch.
    # the parameter pretrained means you will get the empty model or the pretrained successful model.
    # vgg_false = torchvision.models.vgg16(pretrained=False)
    # vgg_true = torchvision.models.vgg16(pretrained=True)

    # because the vgg has trained based on the ImageNet what has 1000 classes.
    # then, we will train based on the cifar10 based on the vgg, so we should
    # modify the output layer from 1000 to 10. we can add linear layer from 1000 to 10.
    # we can also modify the existing layer from 4096 to 10.
    # vgg_true.add_module("add_linear_layer", nn.Linear(1000, 10))
    # vgg_true.classifire.add_module("add_linear", nn.Linear(1000, 10))
    # vgg_false.classifier[6] = nn.Linear(4096, 10)
    # print(vgg_true)

    # store the model
    # method one will store the structure and parameters, method two just stored the parameters.
    # so method2 is recommended.
    # torch.save(vgg_true, "vgg16_method1.pth")
    # torch.save(vgg_false.state_dict(), "vgg16_method2.pth")
    # load the model based on the method1.
    # model = torch.load("vgg16_method1.pth")
    # load the model based on state_dict.
    # model = vgg_false.load_state_dict("vgg16_method2.pths")


    # save your model 
    # cifar10 = Cifar10()
    # torch.save(cifar10, "cifar10.pth")
    # load the model in disk.
    # we have failed to modify our model.
    # model = torch.load("cifar10.pth")
    # model.add_moudle("add_linear", nn.Linear(10, 5))
    # print(model)



    """
    # load_data
    train_data = torchvision.datasets.CIFAR10("../computerVision/data", train = True,
                                              transform=torchvision.transforms.ToTensor(),
                                              download = True)
    
    test_data = torchvision.datasets.CIFAR10("../computerVision/data", train = False,
                                              transform=torchvision.transforms.ToTensor(),
                                              download = True)
    
    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print(f"the length of train data is: {train_data_size}")
    print(f"the length of test data is: {test_data_size}")

    train_dataloader = DataLoader(train_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    # define the model
    cifar10 = Cifar10()
    if torch.cuda.is_available():
        cifar10 = cifar10.cuda()
    # define the loss function.
    loss_function = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_function = loss_function.cuda()
    # define the optimizer
    learning_rate = 1e-2
    optimizer = torch.optim.SGD(cifar10.parameters(), lr=learning_rate)

    # record the training numbers and testing numbers
    total_train_step = 0
    total_test_step = 0

    writer = SummaryWriter("logs")

    # set the parameters about training model.
    # train one epoch you will train all batch. and you should test one time used the trained params each epoch.
    # show the trained loss used tensorboard. total_train_step is the horizontal axis.
    # show the test loss used tensorboard each epoch. total_test_step is the horizontal axis.
    n_epochs = 10
    for i in range(1, n_epochs + 1):
        print(f"------the {i}th training started.---------")

        # started to train
        cifar10.train()
        for data in train_dataloader:
            images, targets = data
            if torch.cuda.is_available():
                images = images.cuda()
                targets = targets.cuda()
            outputs = cifar10(images)
            loss = loss_function(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:
                print(f"the {total_train_step}th training, loss: {loss.item()}")
                # of course, we can imshow the result used tensorboard.
                writer.add_scalar("train_loss", loss.item(), total_train_step)


        
        # test when trained one epoch. and we should calculate the loss for all test data and return it.
        # started to test.
        # started to evaluate, it is dedicated to the dropout layer and so on.
        cifar10.eval()
        total_test_loss = 0
        # just test the result, no need to grad the parameters.
        # we can also calculate the accuracy in test dataset.
        total_accuracy = 0.0
        with torch.no_grad():
            for data in test_dataloader:
                images, targets = data
                if torch.cuda.is_available():
                    images = images.cuda()
                    targets = targets.cuda()
                outputs = cifar10(images)
                loss = loss_function(outputs, targets)
                total_test_loss = total_test_loss + loss
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy
        print(f"the loss in test dataset is: {total_test_loss.item()}")
        print(f"the accuracy in test dataset is: {total_accuracy / test_data_size}")
        writer.add_scalar("test_loss", loss.item(), total_test_step)
        total_test_step = total_test_step + 1


        # store the trained result each epoch.
        torch.save(cifar10, "cifar10{}.pth".format(i))


    writer.close()
    """



    # then, we will test to predict one picture we have trained successfully
    # used gpu
    image_PIL = Image.open("c:/users/80521/desktop/bird1.png")
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    image = transform(image_PIL)
    # first, you should reshape the image
    # because we will use the pretrained model used gpu, so we should set something.
    # when we loaded the model from the local disk.
    # of course, you should import the model or define it in this file.
    # or the pretrained model will not be recognized.
    model = torch.load("../../docs/cifar109.pth", map_location=torch.device('cpu'))
    print(model)
    image = torch.reshape(image, (1, 3, 32, 32))
    # started to test.
    model.eval()
    with torch.no_grad():
        output = model(image)
    print(output.argmax())



    # page158





