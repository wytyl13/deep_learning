'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-07-22 18:39:45
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-07-22 18:39:45
 * @Description: this file is dedicated to running all the class nad method
 * what we have defined in chapter1.
***********************************************************************'''
import torch
import matplotlib.pyplot as plt
import torch.optim as optim

from chapter1 import ResNetGenerator
from chapter1 import Storage
from chapter1 import DataType
from chapter1 import FirstModel
from torch.utils.tensorboard import SummaryWriter





if __name__ == "__main__":
    netG = ResNetGenerator()
    # read the pretrained model horse2zebra what is trained used GAN nueral network.
    # and we should use the load_state_dict method to load the weight and param.
    # only this, the new model will have the same weight and param. what means netG
    # network has acquired all the knowledge the GAN model achieved during training.
    # this is equal to we load resnet101 from torchvision. 
    """ 
    model_path = '../../data/files/horse2zebra_0.3.1.pth'
    model_data = torch.load(model_path)
    netG.load_state_dict(model_data)
    netG.eval() 
    """
    # storage = Storage()
    # storage.basic()

    # dataType = DataType()
    # dataType.basic()

    t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
    t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
    t_c = torch.tensor(t_c)
    t_u = torch.tensor(t_u)
    # we can draw the picture used tensorboard.
    # writer = SummaryWriter("logs")
    # for i in range(t_c.shape[0]):
    #     writer.add_scalar("y = w*x+b", t_c[i], t_u[i])
    # writer.close()

    # we can also draw the picture used matplotlib
    # fig, ax = plt.subplots()
    # plt.scatter(t_c, t_u)
    # ax.set_xlabel("t_c", fontsize=15)
    # ax.set_ylabel("t_u", fontsize=15)
    # plt.show()

    model = FirstModel()
    w = torch.ones(())
    b = torch.zeros(())
    t_p = model.model(t_u, w, b)
    loss = model.loss(t_p, t_c)
    learning_rate = 1e-2
    # loss_rate_of_change_w = model.rate_w(t_c, t_u, w, b)
    # loss_rate_of_change_b = model.rate_b(t_c, t_u, w, b)
    # w = w - learning_rate * loss_rate_of_change_w
    # b = b - learning_rate * loss_rate_of_change_b
    # gradient = model.grad_function(t_u, t_c, t_p, w, b)
    # print(gradient)

    # then we can epoch our gardient function.
    # we can find the loss value has descended enough small. and when we scale the t_u used 0.1, we
    # can use the smaller learning rate for the model.
    # t_un = t_u * 0.1
    # params = model.training_loop(5000, 1e-2, torch.tensor([1.0, 0.0]), t_un, t_c)
    # # then, we can predict the result used the params we have trained successfully.
    # # here, model(t_un, *params) is equivalent to model(t_un, params[0], params[1]).
    # t_p = model.model(t_un, *params)
    # fig = plt.figure(dpi = 300)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.plot(t_u.numpy(), t_p.detach().numpy())
    # plt.plot(t_u.numpy(), t_c.numpy(), 'o')
    # plt.show()

    # then, we can test the autograd in pytorch.
    # t_un = t_u * 0.1
    # params = model.training_loop_autograd(n_epochs=5000,
    #                             learning_rate=1e-2,
    #                             params=torch.tensor([1.0, 0.0], requires_grad=True),
    #                             t_u = t_un,
    #                             t_c = t_c)
    # print(params)

    # test the optim in pytorch.
    # print(dir(optim))
    # t_un = t_u * 0.1
    # params = torch.tensor([1.0, 0.0], requires_grad=True)
    # learning_rate = 1e-2
    # # sgd means the random gradient descent.
    # optimizer = optim.SGD([params], lr=learning_rate)
    # params = model.training_loop_optimizer(5000, optimizer, params, t_un, t_c)
    # print(params)

