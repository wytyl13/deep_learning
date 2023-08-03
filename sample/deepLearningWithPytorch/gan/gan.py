'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-07-24 10:20:39
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-07-24 10:20:39
 * @Description: 
***********************************************************************'''
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Tanh()
        )
    def forward(self, x):
        image = self.main(x)
        image = image.view(-1, 28, 28, 1)
        return image

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator).__init__()
        self.main = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(),
            nn.Linear(521, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.main(x)
        return x




class Gan():
    # of course, we should init the generator and detector.
    # the input of generator is the 100 sizes noise, what meet the normalized distribution.
    # and the output of the generator is to generate one image what the size is similar to the 
    # mnist, 1*28*28.
    # the tanh is more suitable for the generator, and the sigmoid is more suitable for the classifier.
    def __init__(
        self,

    ) -> None:
        pass

    def getData(self):
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(0.5, 0.5)
        ])
        train_ds = torchvision.datasets.MNIST('../../data', train = True, transform = transform, download = True)
        dataloader = torch.utils.data.DataLoader(train_ds, batch_size = 64, shuffle = True)
        return dataloader



if __name__ == "__main__":
    print(torch.__version__)
    gan = Gan()
    dataloader = gan.getData()
    # images, _ = next(iter(dataloader))
    # print(images.shape)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen = Generator().to(device)
    discriminator = Discriminator().to(device)
    discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr = 0.0001)
    generator_optim = torch.optim.Adam(gen.parameters(), lr = 0.0001)

    loss = torch.nn.BCELoss()

