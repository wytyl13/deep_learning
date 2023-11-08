'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-08-02 09:24:17
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-08-02 09:24:17
 * @Description: some utils for deep learning.
 * we will learn some evaluation indicators for deep learning.
 * confusion matrix, recall, precision and MIOU.
 * the confusion matrix:
    the original sample is positive.
        TP: 正确预测为了正样本
        FN: 错误预测为了负样本
    the original sample is negative
        FP: 错误预测为了正样本
        TN: 正确预测为了负样本
    RECALL = TP/(TP+FN) = 正确预测正样本/正样本
    PRECISION = TP/(TP+FP) = 正确预测正样本/预测为正样本
    IOU = 交集/并集
    MIOU = TP/(TP+FP+FN) = 平均IOU suitable for the image segmentation.
    MIOU = 正确预测为正样本/

    we can label the image used labelimg or labelme.


***********************************************************************'''

from PIL import Image, ImageDraw
import os
import json
import numpy as np
import cv2
import torch

CLASS_NAMES = ['background', 'dog', 'cat', 'person', 'horse']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WEIGHT_PATH_VOC = "../../../docs/unet_voc.pth"
WEIGHT_PATH_SELF = "../../../docs/unet_self.pth"
DATA_PATH_VOC = "../../../data/VOC"
DATA_PATH_SELF = "../../../data/unet_image"
SAVE_TRAINING_IMAGE_VOC = "../../../data/unet_image/training_image/voc"
SAVE_TRAINING_IMAGE_SELF = "../../../data/unet_image/training_image/self"
VOC = True


def getPalette(img_path):
    """
    获取图片调色板信息
    """
    img = Image.open(img_path)
    palette = img.getpalette()
    return palette

def tensorToPImage(mask_np, palette):
    """
    输入：图片的nparray 和 调色板信息
    输出：对应位图
    """
    img = np.uint8(mask_np)
    img = Image.fromarray(img)
    img.putpalette(palette)
    return img


def scale_image_RGB(path, size = (256, 256)):
    image = Image.open(path)
    max_len = max(image.size)
    mask = Image.new('RGB', (max_len, max_len), (0, 0, 0))
    # paste the image to the left upper of the mask.
    mask.paste(image, (0, 0))
    mask = mask.resize(size)
    return mask

def scale_image_P(path, size=(256, 256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('P', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask

def scale_image_mask(path, size=(256, 256)):
    img = Image.open(path)
    p = img.getpalette()
    temp = max(img.size)
    mask = Image.new('P', (temp, temp))
    mask.putpalette(p)
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask

'''
 * @Author: weiyutao
 * @Date: 2023-08-02 16:14:29
 * @Parameters: 
 * @Return: 
 * @Description: this function is dedicated to transform the json file what make from
 labelme tool. they are some label point for one image, we will generate one mask
 image based on the json file and original image used this function as follow. this function can just
 transform the original image and correspond json file to the single channel image. and the gray value
 corresponding to its class index.我们这里使用的是单通道的灰度图像来对分割图做处理，比如是将标记好
 的xml文件转换为灰度图像，灰度值对应的就是标记的物体的索引值。这样我们可以在训练的时候对对应的灰度图
 和原始图去定义交叉熵损失函数，也就是多分类的损失函数，最后训练出来的参数会使得生成的分割图像向对应的灰度值去靠近。
 然后如果我们想要展示生成的分割图像，因为他是灰度图像，而且灰度值对应的是物体的索引，所以直接展示是黑色的。
 我们可以使用缩放灰度的方式对每一个像素点乘以20或者某一个固定的值，然后再展示该灰度图像即可。
 当然我们也可以使用Image类的调色板方法，调色版的方法就是尽量使用单通的图像去表示全部颜色，也就是我们的
 单通道图像中的每一个灰度值都可以最终转换为彩色图像。当然，如果我们之前已经定义好了调色版，那么
 我们可以直接使用plt.imshow去展示该单通道图像对应的调色版图像。我们这里对VOC数据集使用了调色版的方式
 因为它的分类数量比较多，而我们对我们自定义的训练数据使用了单通道灰度图像的表示方式。
 '''
def make_mask(image_dir, save_dir):
    data = os.listdir(image_dir)
    json_files = []
    # store all the json file in image_dir used variable temp_data
    for i in data:
        if i.split('.')[1] == 'json':
            json_files.append(i)
        else:
            continue

    for json_file in json_files:
        # open the json file and read it.
        json_content = json.load(open(os.path.join(image_dir, json_file), 'r'))
        shapes = json_content['shapes']
        # init the mask image what has the same size with the original image.
        mask = Image.new('P', Image.open(
            os.path.join(image_dir, json_file.replace('json', 'png'))).size)
        # because one image could have many objetc, so we should for circle 
        # the shapes.
        for shape in shapes:
            label = shape['label']
            points = shape['points']
            # transform to tuple type.
            points = tuple(tuple(i) for i in points)
            mask_draw = ImageDraw.Draw(mask)
            # draw the polygon used each points for each object.
            mask_draw.polygon(points, fill=CLASS_NAMES.index(label))
        # store the mask image.
        # show the image, you should *255, because we have define the 
        # mask used 0 + index.
        mask = np.array(mask)
        cv2.imwrite(os.path.join(save_dir, json_file.replace('json', 'png')), mask)
        # print(set(mask.reshape(-1).tolist()))
        # mask = np.array(mask) * 255
        # print(type(mask))
        # print(np.max(mask))
        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)
        # this will store the mask image, what the background is zero piexl, and
        # the other object is 1, 2, 3, 4 based on the index position.
        # mask.save(os.path.join(save_dir, json_file.replace('json', 'png')))


# show all the object label for one mask image we have generated based on the above code.
# we can not change the pixel based on the label index.
def visual_label_image(image_path):
    image = Image.open(image_path)
    # image = cv2.imread(image_path, 0)
    image = np.array(image)
    # flatten the image first, then to list, get the set at last.
    print(set(image.reshape(-1).tolist()))
    image = np.array(image) * 40
    print(set(image.reshape(-1).tolist()))
    cv2.imshow("image", image)
    cv2.waitKey(0)


def merge_dic(dict1, dict2):
    result_dict = {}
    for k, v in dict1.items():
        for m, n in dict2.items():
            if k == m:
                result_dict[k] = []
                result_dict[k].append(dict1[k])
                result_dict[k].append(dict2[k])
                dict1[k] = result_dict[k]
                dict2[k] = result_dict[k]
            else:
                result_dict[k] = dict1[k]
                result_dict[m] = dict2[m]
    return result_dict



if __name__ == "__main__":
    # generate the corresponding json file based on labelme. 
    # then generate corrsponding mask image based on the script code.
    # notice you should put the orginal image and json file in one dictory.
    # and they have the same name. then, we will generate the segmentation image
    # to the path that based on the second param.
    # make_mask("../../../data/unet_image/JPEGImages", "../../../data/unet_image/SegmentationClass")
    visual_label_image("../../../data/unet_image/SegmentationClass/catdog.png")
    # dict1 = {'a': 1, 'b': 2}
    # dict2 = {'a': 3, 'b': 4, 'c': 8}
    # dict3 = {}
    # result_dict = merge_dic(dict3, dict2)
    # print(result_dict)