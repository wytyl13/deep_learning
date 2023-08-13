'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-08-10 17:08:36
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-08-10 17:08:36
 * @Description: 
***********************************************************************'''
import torch


'''
 * @Author: weiyutao
 * @Date: 2023-08-10 17:09:24
 * @Parameters: box: [x1, y1, x2, y2]分别对应的左上角和右下角的坐标点。
 * @Return: 
 * @Description: 
 * 首先我们来定义iou的计算方式，交并比，交集有很多种方式，比如最常用的就是两个框存在交集。
 * 特殊的比如一个框包含另一个框。但是无论如何我们都可以用下面这种办法来计算两个框的交集面积。
 * 我们可以使用两个点来确定两个框交集的面积，两个框左上角的两个点选靠下的一个，两个框右下角的点
 * 选靠上的一个。这两个点构成的面积就是交集。

 '''
def iou(box, boxes, isMin = False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # 交集
    # 这里交集计算可以涵盖任何的交集方式，只要计算出两个框左上角点x的最大值和y的最大值，就是交集矩形的左上角的坐标
    # 只要计算出两个框右下角点x的最小值和y的最小值，就是交集矩形的右下角的坐标。
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])

    w, h = torch.maximum(torch.tensor([0]), x2 - x1), torch.maximum(torch.tensor([0]), y2 - y1)
    intersection_area = w * h
    if isMin:
        return intersection_area / torch.min(box_area, boxes_areas)
    else:
        return intersection_area / (box_area + boxes_areas - intersection_area)

'''
 * @Author: weiyutao
 * @Date: 2023-08-10 17:32:03
 * @Parameters: boxes: [[confidence, x1, y1, x2, y2], ...]
 * @Return: 
 * @Description: 
 '''
def nms(boxes, threshold_value, isMin = False):
    # 首先按照置信度对所有的box进行排序。
    new_boxes = boxes[boxes[:, 0].argsort(descending=True)]
    keep_boxes = []
    while len(new_boxes) > 0:
        box_ = new_boxes[0]
        print(box_)
        keep_boxes.append(box_)
        if len(new_boxes) == 1:
            break
        boxes_ = new_boxes[1:]
        # 小于阈值的才会在本次比较中保留下来，然后找到小于阈值的索引，重新定义保留下来的
        # 所有boxes为new_boxes. 然后继续循环即可。
        new_boxes = boxes_[torch.where(iou(box_[1:], boxes_[:, 1:], isMin) < threshold_value)]
    return torch.stack(keep_boxes)





if __name__ == "__main__":
    box = torch.tensor([0, 0, 4, 4])
    boxes = torch.tensor([[0.5, 1, 1, 10, 10], [0.9, 1, 2, 11, 11], [0.4, 8, 8, 12, 12]])
    # iou = iou(box, boxes)
    print(nms(boxes, 0.5)) 