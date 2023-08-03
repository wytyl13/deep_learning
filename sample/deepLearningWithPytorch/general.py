'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-07-08 17:51:44
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-07-08 17:51:44
 * @Description: main neural network. convolusion neural network, recurrent neural network
 and graph neural network, you can name them used CNN, RNN AND GNN. the CNN is dedicated
 to the image process, the RNN is dedicated to using for the image descripe, generate, machine
 translation, speech recognition and video tags.
 the traditional feature extract algorithm need to code the metho by ourselves, then, the deep
 learning need not to do this by ourselves, we just need to define the convolusion neural network
 and the trained process will learn the best convolusion kernel to extract the feature in image.
***********************************************************************'''


from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


image_path = "c:/users/80521/desktop/bird.png"
image = Image.open(image_path)
# create one tool that can transform one image to tensor.
tool = transforms.ToTensor()
tensor_image = tool(image)


writer = SummaryWriter("logs")
for i in range(100):
    writer.add_scalar("y = x", i, i)
writer.close()