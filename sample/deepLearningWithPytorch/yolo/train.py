from torch.utils.data import Dataset
from PIL import Image
import os
from utils import DATA_FILE_PATH
from utils import IMAGE_DIR_PATH
import numpy as np
import math
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


from utils import scale_image_416
import config
from config import DATA_WIDTH
from config import DATA_HEIGHT
from utils import one_hot
from utils import WEIGHT_PATH
from net import Yolov3
from utils import DEVICE


class MyDataset(Dataset):

    def __init__(self) -> None:
        super(MyDataset, self).__init__()
        f = open(DATA_FILE_PATH, 'r')
        self.dataset = f.readlines()

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        labels = {}
        data = self.dataset[index]
        data_value = data.split()
        file_name = data_value[0]
        image_path = os.path.join(IMAGE_DIR_PATH, file_name)
        # get all the data except the first index file name.
        data_box_cls = data_value[1:]
        # we should transform the variable to value type. then we can
        # split it.
        data_box_cls_value = np.array([float(x) for x in data_box_cls])
        boxes = np.split(data_box_cls_value, len(data_box_cls)//5)

        # resize the image.
        scale_rate, image_data = scale_image_416(image_path)

        # 
        for feature_size, anchors in config.ANCHORS.items():
            # define the output shape.
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + config.CLASS_NUM))
            # then, we should calculate the offset based on the different size.
            # we have three size, 13, 26, 52. the original image size is 416.
            # then, we should calculate the offset about the center point of the 13 size feature
            # map to the 416 size image. that's all. just like 13*13, each element in 13 feature
            # map will represent 23 element in the original image. so 23 is the scale for 13 to 416.
            # just like the x axis of the center point of the object in 416 image is 112, then the corresponding
            # x axis of the center point for feature map is 112/32 = 3.5. then the 3.5 can be expressed
            # as the third element and 32*0.5=16 element. so the x axis location is this. y axis is
            # similar. and we should consider the scale rate from original size to 416.
            # this scale_rate we can got by the scale_image_416 function.
            for box in boxes:
                cls, cx, cy, w, h = box
                # consider the scale_rate after scale_image_416 function
                cx, cy, w, h = cx * scale_rate, cy * scale_rate, w * scale_rate, h * scale_rate
                # cx_offset, cx_index = math.modf(cx / (DATA_WIDTH / feature_size))
                # calculate the offset what involved integer and decimal.
                cx_offset_integer, cx_offset_decimal = math.modf(cx * feature_size / DATA_WIDTH)
                cy_offset_integer, cy_offset_decimal = math.modf(cy * feature_size / DATA_HEIGHT)

                # calculate the scale for the three anchors and w, h.
                for i, anchor in enumerate(anchors):
                    anchor_area = config.ANCHORS_AREA[feature_size][i]
                    p_w, p_h = w / anchor[0], h / anchor[0]
                    p_area = w * h
                    iou = min(p_area, anchor_area) / max(p_area, anchor_area)

                    # e^p_w = w / w^, because in order to avoid the p_w is negative value.
                    # so we can get p_w = log(w / w^), w is the actual width for one box in original  image.
                    # w^ is the designed width. it is defined in config files. it is the experience data.
                    # p_w is equal to log(w / w^), * means the changable variable.
                    if labels[feature_size][int(cy_offset_decimal), int(cx_offset_decimal), i][0] < iou:
                        labels[feature_size][int(cy_offset_decimal), int(cx_offset_decimal), i] = \
                        np.array([iou, cx_offset_integer, cy_offset_integer, np.log(p_h), np.log(p_w), *one_hot(config.CLASS_NUM, int(cls))])


        return labels[13], labels[26], labels[52], image_data


        # print(boxes)
        # print(image_data.shape)
        # print(scale_rate)
        # then we should resize the image from 480*364 to 416*416. and ensure
        # the image after resizing will not change the scale.

# we have defined the dataset class, then we can go on training the data.
'''
 * @Author: weiyutao
 * @Date: 2023-08-05 10:37:53
 * @Parameters: ouput is the out result by the yolov3 net. its shape is (N, 24, 13, 13) or
 * 26, 52. because the dataloader data shape is different from the output of net.
 * so we should reshape them to same before calculate the loss value. and you should divide
 * the channels what involved confidence, x, y, width, height, and one_hot class. because you
 * should calculate the corresponding loss for each of them. target is the data we have loaded.
 * the shape of target is (N, w, h, 3, 8), 3 is the scale numbers, 8 is 5+3. so the begin we should
 * transfor the shape of out from (N, 8, 13, 13) to (N, 13, 13, 3, 24/3=8) 
 * @Return: 
 * @Description: 
 '''
def loss_function(output, target, alpha):
    # reshape from(N, channels, 13, 13) to (N, 13, 13, channels)
    output = output.permute(0, 2, 3, 1)
    # reshape from (N, 13, 13, channels) to (N, 13, 13, 3, channels/3)
    # this case is (N, 13, 13, 3, 8), 8=5+3, 5 is the num of c, x, y, w, h.
    # 8 is the class numbers.
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
    # then, we should find the last dimension for the target. its last dimension is 
    # 8 what involved c, x, y, w, h, one_hot class num. we handled the confidence first.
    # the confidence is the first index for the last dimension of target.
    # if the confidence > 0, then there are the object has been detected. or there are not
    # the object has been detected.
    is_object = target[..., 0] > 0
    no_object = target[..., 0] == 0

    # then, the confidence loss function is one binary classifier problem.
    # so we can use BCELoss to define its loss function.
    # but notice, we should use sigmoid function to normalize the output to 0~1.
    # but the label is normal confidence value.
    confidence_loss_function = nn.BCELoss()
    confidence_loss = confidence_loss_function(torch.sigmoid(output[..., 0]), target[..., 0])


    # then, we should define the loss function about x, y, w, h. it is one regression problem.
    # so we can use mean standard error what is MSELoss in torch.nn
    # of course, we should just calculate the is_object index. no_object index should
    # not be considered. the x, y, w, h is from index 1 to index 4.
    # ... in list means the last dimension.
    box_loss_function = nn.MSELoss()
    box_loss = box_loss_function(output[is_object][..., 1:5], target[is_object][..., 1:5])

    # then we will define the loss function about classifier. the index is from 5:
    # we can use the multiple classifier cross entropy loss.
    # but you should notice, the crossentropyloss in nn involved the one_hot operation for the second param.
    # but we have operated the one_hot in dataloader. so we should reduction it to non one_hot.
    # output need not to handle.
    classifier_loss_function = nn.CrossEntropyLoss()
    classifier_loss = classifier_loss_function(output[is_object][..., 5:], torch.argmax(target[is_object][..., 5:], dim=1, keepdim=True).squeeze(dim=1))

    # then, we should calculate all the loss value at last. but you should notice the positive object in
    # one image is less than the negative object. so it measn the confidence loss value will be very
    # small. and the box loss and classifier is not been influenced, because they is calculated based on
    # the positive object. so then in order to balance this unequal loss, we should define the weight
    # before we generate the total loss value. we can pass one weight param. we used alpha = 0.5 here.
    loss = alpha * confidence_loss + (1 - alpha) * 0.5 * box_loss + (1 - alpha) * 0.5 * classifier_loss
    # the weight is 0.5, 0.25, 0.25. or [alpha, (1-alpha)*0.5, (1-alpha)*0.5]
    return loss



if __name__ == "__main__":
    # TEST THE DATASET AND NET WORKING.
    # dataset = MyDataset()
    # # print(dataset[0].shape)
    # # print(len(dataset))
    # print(dataset[0][3].shape)
    # print(dataset[0][0].shape)
    # print(dataset[0][1].shape)
    # print(dataset[0][2].shape)

    writer = SummaryWriter('logs')

    # STARTED TO TRAIN.
    dataset = MyDataset()
    # shuffle means if disrupted the data.
    train_loader = DataLoader(dataset, batch_size=5, shuffle=True)
    net = Yolov3().to(DEVICE)
    if os.path.exists(WEIGHT_PATH):
        net.load_state_dict(torch.load(WEIGHT_PATH))
    
    # start mode.
    net.train()
    optimizer = torch.optim.Adam(net.parameters())
    epoch = 0
    index = 0
    while True:
        for target_13, target_26, target_52, image_data in train_loader:
            # notice the shape for target_13, 26, 52.
            # its shape is (13, 13, 3, 8) ,8 is equal to 5+class_num.
            # this case class_num is 3. 3 is three scale designed box for each size.
            target_13, target_26, target_52, image_data = target_13.to(DEVICE), \
            target_26.to(DEVICE), target_52.to(DEVICE), image_data.to(DEVICE)

            # you should notice the output shape. it is (N, 8, 13, 13)
            out_13, out_26, out_52 = net(image_data)
            
            # calculate the loss value. loss function invovled all parameters
            # just like confidence, x, y, w, h and class. we can divide it into
            # 13, 26, and 52 loss value.
            loss_13 = loss_function(out_13.float(), target_13.float(), 0.6)
            loss_26 = loss_function(out_26.float(), target_26.float(), 0.6)
            loss_52 = loss_function(out_52.float(), target_52.float(), 0.6)
            loss = loss_13 + loss_26 + loss_52

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'loss {epoch} == {index}', loss.item())
            writer.add_scalar('train_loss', loss, index)
            index += 1
        torch.save(net.state_dict(), f'data/net{epoch}.pt')
        print(f'{epoch} save successful')
        epoch += 1
            
