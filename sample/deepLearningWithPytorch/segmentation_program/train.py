import torch
from torch import nn, optim



from torch.utils.data import DataLoader
import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt

from model_classic_unet import UNet
from torchvision.utils import save_image
from utils import CLASS_NAMES
from utils import WEIGHT_PATH_VOC
from utils import WEIGHT_PATH_SELF
from utils import DATA_PATH_VOC
from utils import DATA_PATH_SELF
from utils import SAVE_TRAINING_IMAGE_VOC
from utils import SAVE_TRAINING_IMAGE_SELF
from utils import DEVICE
from utils import VOC
from datasets import MyDataset
from utils import getPalette
from utils import tensorToPImage

    
    

if __name__ == "__main__":
    # notice the num_class is equal to your class number + backgound.
    # if you want to train your data, because it is gray image in this case,
    # and it is the multipy classifier problem. so you should define the
    # out channel used the numers of CLASS_NAMES
    num_classes = len(CLASS_NAMES)

    # if you want to train VOC, because it is the color image and
    # it is the binary classifier problem. so you should define the num_class
    # used 3 as the output channel. and you should change the correspond
    # loss function and scale image and the last layer of the unet.
    # because only the binary classifier problem need to use the sigmoid.
    # the multipy classifier problem used the softmax.
    if VOC:
        num_classes = 21
        data_loader = DataLoader(MyDataset(DATA_PATH_VOC, num_classes), batch_size=1, shuffle=True)
    else:
        data_loader = DataLoader(MyDataset(DATA_PATH_SELF, num_classes), batch_size=1, shuffle=True)
    net = UNet(num_classes).to(DEVICE)
    if VOC:
        if os.path.exists(WEIGHT_PATH_VOC):
            net.load_state_dict(torch.load(WEIGHT_PATH_VOC))
            print("successful load weight file")
        else:
            print("failer to load weight file")
    else:
        if os.path.exists(WEIGHT_PATH_SELF):
            net.load_state_dict(torch.load(WEIGHT_PATH_SELF))
            print("successful load weight file")
        else:
            print("failer to load weight file")
    
    optimizer = optim.Adam(net.parameters())
    # notice the crossentropyloss function has involved onehot code
    # for the param.
    # this is crossEntropyLoss, it is dedicated to multipy classifier.
    # so it is dedicated to our training data in this case, if you want
    # to train the voc dataset, you should use BCELoss function. because
    # voc is one binary multipy problem.

    # train your data, please open this code and close the next code
    
    ###############################################################
    loss_function = nn.CrossEntropyLoss()
    # train the voc data, please open this code and close the former code.
    # if VOC:
    #     loss_function = nn.BCELoss()
    ###############################################################
    epoch = 1
    while True:
        # tqdm is one expansion progress bar.
        for i, (image, segment_image) in enumerate(tqdm.tqdm(data_loader)):
            image, segment_image = image.to(DEVICE), segment_image.to(DEVICE)
            out_image = net(image)

            # IF YOU READ COLOR IMAGE. YOU SHOULD PERMUTE THE CHANNELS ORDER.
            # if VOC:
            #     segment_image = segment_image.permute(0, 3, 1, 2)
            # notice the out_image is num_classes*width*height
            # and the segment_image is gray image, so it is 1*width*height.
            # so if you used the simple loss function, it will be error.
            # but we have used the crossentropyloss function, it will
            # execute the one hot code first.
            # if VOC:
            #     train_loss = loss_function(out_image, segment_image)
            # else:
            train_loss = loss_function(out_image, segment_image.long())  

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print(f"{epoch}-{i}-train_loss ===>> {train_loss.item()}")
            
            

            # contact three image. orginal image, segment_image, and predict image.
            # notice, if you have generated one single channel image, and
            # the pixel is generated based on the label index. you should
            # multi 255 for each pixel. or you will not fail to show the image.
            # notice, if your segmentation image and predict image are all
            # gray image, but your train image is color, you can contact them.
            # you can just contact these two gray image.
                if VOC:
                    image = torch.stack([torch.unsqueeze(segment_image[0], 0), torch.argmax(out_image[0], dim=0).unsqueeze(0)], dim=0)
                    # original_image = image[0].permute(1, 2, 0)
                    # print(image.shape)
                    # print(segment_image[0].shape)
                    # print(out_image[0].shape)
                    # plt.figure()
                    # plt.subplot(1, 3, 1)
                    # plt.imshow(original_image)
                    # plt.subplot(1, 3, 2)
                    # plt.imshow(segment_image[0])
                    # plt.subplot(1, 3, 3)
                    # plt.imshow(out_image[0].argmax(0))
                    # plt.show()
                else:
                    image = torch.stack([
                        torch.unsqueeze(segment_image[0], 0) * 255, 
                        torch.argmax(out_image[0], dim=0).unsqueeze(0) * 255
                        ], dim=0)
            if i % 1000 == 0:
                if VOC:
                    save_image(image, f'{SAVE_TRAINING_IMAGE_VOC}/{epoch}_{i}.png')
                else:
                    save_image(image, f'{SAVE_TRAINING_IMAGE_SELF}/{epoch}_{i}.png')
        if epoch % 20 == 0:
            if VOC:
                torch.save(net.state_dict(), WEIGHT_PATH_VOC)
            else:
                torch.save(net.state_dict(), WEIGHT_PATH_SELF)
        epoch += 1




























