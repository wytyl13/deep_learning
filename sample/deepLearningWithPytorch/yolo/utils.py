'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-08-04 15:27:11
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-08-04 15:27:11
 * @Description: this file is some tools for yolo model. involved train and test. and predict
***********************************************************************'''
import xml.etree.cElementTree as et
import os
import math
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch
from PIL import ImageDraw


from config import DATA_WIDTH, DATA_HEIGHT

transform = transforms.Compose([
    transforms.ToTensor()
])


XML_DIR_PATH = "data/image_voc"
DATA_FILE_PATH = 'data/data.txt'
IMAGE_DIR_PATH = 'data/images'
WEIGHT_PATH = 'data/net80.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_DICT = {
    'person': 0,
    'horse': 1,
    'bicycle': 2
}


def make_image_data(path):
    img=Image.open(path)
    w,h=img.size[0],img.size[1]
    temp=max(h,w)
    mask=Image.new('RGB',(temp,temp),(0,0,0))
    mask.paste(img,(0,0))
    return mask


def iou(box, boxes, mode="inter"):
    cx, cy, w, h = box[2], box[3], box[4], box[5]
    cxs, cys, ws, hs = boxes[:, 2], boxes[:, 3], boxes[:, 4], boxes[:, 5]

    box_area = w * h # 最小面积
    boxes_area = ws * hs # 最大面积

    _x1, _x2, _y1, _y2 = cx - w/2, cx + w/2, cy - h/2, cy + h/2
    _xx1, _xx2, _yy1, _yy2 = cxs - ws / 2, cxs + ws / 2, cys - hs / 2, cys + hs / 2

    xx1 = torch.maximum(_x1, _xx1) # 左上角   最大值
    yy1 = torch.maximum(_y1, _yy1) # 左上角   最大值
    xx2 = torch.minimum(_x2, _xx2) # 右下角  最小值
    yy2 = torch.minimum(_y2, _yy2) # 右下角  最小值

    # 将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。
    w = torch.clamp(xx2 - xx1, min=0) # ★夹
    h = torch.clamp(yy2 - yy1, min=0)

    inter = w * h

    if mode == 'inter':
        return inter / (box_area + boxes_area - inter) #交集除以并集
    elif mode == 'min':
        return inter / torch.min(box_area, boxes_area)

def nms(boxes, thresh, mode='inter'):
    args = boxes[:, 1].argsort(descending=True)
    sort_boxes = boxes[args]
    keep_boxes = []

    while len(sort_boxes) > 0:
        _box = sort_boxes[0]
        keep_boxes.append(_box)

        if len(sort_boxes) > 1:
            _boxes = sort_boxes[1:]
            # print(_clses.shape)
            # print(_cls.shape)
            # print(mask.shape, "-------------------")
            # print(_boxes)
            # print(_boxes.shape)

            _iou = iou(_box, _boxes, mode)
            sort_boxes=_boxes[_iou< thresh]

        else:
            break

    return keep_boxes



def make_xml_txt(xml_dir_path, image_dir_path):
    xml_filenames = os.listdir(xml_dir_path)
    with open('data/data.txt', 'a') as f:
        f.truncate(0)
        for xml_filename in xml_filenames:
            xml_file_path = os.path.join(xml_dir_path, xml_filename)
            # tree_content is the xml file name based on tree structure.
            # root is the root label for the xml content. we can use root.find to find
            # all the content in xml file based on the label name.
            tree_content = et.parse(xml_file_path)
            root = tree_content.getroot()
            # name is a list that has one index.
            file_name = root.find('filename').text
            class_names = root.findall('object/name')
            # we can get bound box information by selecting the object/bndbox label.
            boxes_information = root.findall('object/bndbox')
            max_length_image = max(Image.open(os.path.join(image_dir_path, file_name)).size)
            data = []
            data.append(file_name)

            # test draw rectangle onto the scale image. what is the input of the net.
            """
            image_path = os.path.join(image_dir_path, file_name)
            # scale_rate, image_data = scale_image_416(image_path)
            # image_pil = transforms.ToPILImage()(image_data)
            # image_pil.show(file_name)
            # print(image_path)
            """
            
            for class_name, box_information in zip(class_names, boxes_information):
                class_index = CLASS_DICT[class_name.text]
                # cx = (min_X + max_X) / 2
                # notice the data in box is min_x, min_y, max_x and max_y.
                cx, cy = math.floor((int(box_information[0].text) + int(box_information[2].text)) / 2), \
                        math.floor((int(box_information[1].text) + int(box_information[3].text)) / 2)
                w, h = (int(box_information[2].text) - int(box_information[0].text)), \
                        (int(box_information[3].text) - int(box_information[1].text))
                # obj stored all information about class index, cx, cy, width and height.
                # notice the scale. because we have scale the orginal image from max_length_image to 416.
                # so the scale will be 416 / max_length_imag. and we need to the scale_rate in dataset.
                obj = f"{class_index},{math.floor(cx * 416 / max_length_image)},{math.floor(cy * 416 / max_length_image)},{math.floor(w * 416 / max_length_image)},{math.floor(h * 416 / max_length_image)}"
                data.append(obj)

                # test the image rectangle after scale resize.
                """                 
                x0 = math.floor(cx * 416 / max_length_image) - math.floor(w * 416 / max_length_image) / 2
                y0 = math.floor(cy * 416 / max_length_image) - math.floor(h * 416 / max_length_image) / 2
                x1 = x0 + math.floor(w * 416 / max_length_image)
                y1 = y0 + math.floor(h * 416 / max_length_image)
                ImageDraw.Draw(image_pil).rectangle([x0, y0, x1, y1], fill=None, outline='red', width=2) 
                """
                # so the data we will store into the data.txt is image_name, [class_index, cx, cy, w, h].recycle.
                # because one image might has many boxes, so it will have many box list.  

            # test the image rectangle after scale resize.
            # image_pil.show(file_name)

            str = ''
            for i in data:
                str = str + i + ','
            str = str.replace(',', ' ').strip()
            f.write(str + '\n')
    f.close()



def scale_image_416(image_path):
    image = Image.open(image_path)
    w, h = image.size
    max_length = max(w, h)
    mask_image = Image.new(mode='RGB', size=(max_length, max_length), color=(0, 0, 0))
    mask_image.paste(image, (0, 0))
    mask_image = mask_image.resize((DATA_WIDTH, DATA_HEIGHT))
    # notice, transforms.totensor has normalized the original image_data.
    image_data = transform(mask_image)
    scale_rate = DATA_WIDTH / max_length
    return scale_rate, image_data



def one_hot(class_num, i):
    b = np.zeros(class_num)
    b[i] = 1
    return b



if __name__ == "__main__":
    make_xml_txt(XML_DIR_PATH, IMAGE_DIR_PATH)
    # image_data = scale_image_416('data/images/000017.jpg')
    # print(image_data)
    # Image show the image.
    # mask_image.show()

    # plt show the image.
    # mask = np.array(mask_image)
    # plt.imshow(mask)
    # plt.show()

