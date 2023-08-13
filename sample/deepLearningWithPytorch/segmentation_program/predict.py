import os
import numpy as np
import torch
import cv2


from train import *
from model_classic_unet import UNet
from utils import scale_image_P
from utils import scale_image_RGB
from utils import CLASS_NAMES
from utils import DEVICE
from utils import WEIGHT_PATH
from datasets import transform

if __name__ == "__main__":
    # notice the num_classes must be similar to the trained model.
    num_classes = len(CLASS_NAMES)
    net = UNet(num_classes).to(DEVICE)
    if os.path.exists(WEIGHT_PATH):
        net.load_state_dict(torch.load(WEIGHT_PATH))
        print('successful load')
    else:
        print("fail to load weight file")
    
    _input = input('please input the JPEGImages path:')
    image_full_name = _input.split('/')[-1]
    image_name = image_full_name.split('.')[0]
    image = scale_image_RGB(_input)
    image_data = transform(image).to(DEVICE)
    image_data = torch.unsqueeze(image_data, dim=0)
    net.eval()
    out = net(image_data)
    out = torch.argmax(out, dim=1)
    out = torch.squeeze(out, dim=0)
    out = out.unsqueeze(dim=0)
    print(set((out).reshape(-1).tolist()))
    print(out.shape)
    out=(out).permute((1,2,0)).cpu().detach().numpy()
    print(out.shape)
    cv2.imwrite(f'../../../data/unet_image/predict_result/{image_name}.png',out)
    # if you want to imshow the array used cv2, you should use
    # the dimension(width, height, channel)
    cv2.imshow(image_name, out*255.0)
    cv2.waitKey(0)
