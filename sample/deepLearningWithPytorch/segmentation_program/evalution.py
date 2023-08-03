'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-08-03 11:17:40
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-08-03 11:17:40
 * @Description: of course, we should evalution the model based some indicators.
 just like miou, recall and precision and so on.
***********************************************************************'''
from PIL import Image
import numpy as np
import os
from sklearn.metrics import confusion_matrix


from utils import merge_dic
from utils import CLASS_NAMES


'''
 * @Author: weiyutao
 * @Date: 2023-08-03 11:18:53
 * @Parameters: 
 * @Return: 
 * @Description: first, we should resize the label image what is segmentation image
 in segmentation application. resize the predict image. in order to avoid change the 
 scale for the image, we should use the max length of the image to create one rectangular
 image. then paste the target image onto the left upper of the image. then resize it.
 we will not change the scale for the image what you want to resize.
 '''
def scale_image_segmentation(path, size=(256, 256)):
    image = Image.open(path)
    max_length = max(image.size)
    mask = Image.new('P', (max_length, max_length))
    mask.paste(image, (0, 0))
    mask = mask.resize(size)
    mask = np.array(mask)
    # reset to binary image for all object.
    mask[mask != 255] = 0
    mask[mask == 255] = 1
    mask = Image.fromarray(mask)
    return mask


def scale_image_predict(path, size=(256, 256)):
    image = Image.open(path)
    max_length = max(image.size)
    mask = Image.new('P', (max_length, max_length))
    mask.paste(image, (0, 0))
    mask = mask.resize(size)
    mask = np.array(mask)
    mask = Image.fromarray(mask)
    return mask


def compute_iou(segmentation_predict, segmentation_label, num_classes):
    ious = []
    for c in range(num_classes):
        predict_inds = segmentation_predict == c
        label_inds = segmentation_label == c
        # calculate the area about intersection and union.
        intersection = np.logical_and(predict_inds, label_inds).sum()
        union = np.logical_or(predict_inds, label_inds).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(union))
    
    return ious


def compute_miou(segmentation_predicts, segmentation_labels, num_classes):
    ious = []
    for i in range(len(segmentation_predicts)):
        ious.append(compute_iou(segmentation_predicts[i], segmentation_labels[i], num_classes))
    ious = np.array(ious, dtype=np.float32)
    miou = np.nanmean(ious, axis=0)
    return miou

def compute_indicators(segmentation_path, predict_path):
    predict_image = scale_image_predict(predict_path)
    segmentation_image = scale_image_predict(segmentation_path)
    l, p = np.array(segmentation_image).reshape(-1), np.array(predict_image).reshape(-1)
    segmentation_list = list(set(l.tolist()))
    out = confusion_matrix(l, p, labels=segmentation_list)
    r, c = out.shape
    iou = 0
    recall = {}
    precision = {}
    for i in range(r):
        class_name = CLASS_NAMES[i]
        TP = out[i, i]
        temp = np.concatenate((out[0:i, :], out[i+1:, :]), axis=0)
        FP = np.sum(temp, axis=0)[i]
        temp2 = np.concatenate((out[:, 0:i], out[:, i+1:]), axis=1)
        FN = np.sum(temp2, axis=1)[i]
        # notice TN involved all the actual label is negative, and 
        # predict to as negative sample. notice no matter the negative sample
        # is predict true or false. just like we are calculate the class 1. then
        # we should consider the actual class is 2 or 3, and predict to as 2 or 3
        # as the part of TN. so TN = temp2.reshape(-1).sum() - FN. 
        TN = temp2.reshape(-1).sum() - FN
        iou += TP / (TP + FP + FN)
        # recall means the predic true for positive sample divide into
        # all the positive sample numbers.
        recall[class_name] = TP / (TP + FN)
        # precision means the predict true for positive sample
        # divide into all predict positive samples.
        # NOTICE, if TP and FP are both zero, the precision will returen nan.
        # so it means this class is not be recognized.
        precision[class_name] = TP / (TP + FP)
        # print(TP, FP)
        # recall 意思是正确预测正样本数在所有正样本数中的占比。
        # precision意思是正确预测正样本数在所有预测为正样本数中的占比。
        # iou意思是正确预测正样本数在在所有正样本数+错误预测为正样本数中的占比。
    miou = iou / r
    return recall, precision, miou


# give the predict image dir path, calculate the recall, precision and miou.
def compute_indicators_dir(segmentation_dir_path, predict_dir_path):
    recall = {}
    precision = {}
    miou = 0
    result = {}
    for pred_image in os.listdir(predict_dir_path):
        segmentation_path = os.path.join(segmentation_dir_path, pred_image)
        predict_path = os.path.join(predict_dir_path, pred_image)
        recall_, precision_, miou_ = compute_indicators(segmentation_path, predict_path)
        miou += miou_
        if len(recall) == 0:
            recall = recall_
            precision = precision_
            continue
        recall = merge_dic(recall, recall_)
        precision = merge_dic(precision, precision_)
    for key, value in recall.items():
        recall[key] = np.mean(value)
    for key, value in precision.items():
        precision[key] = np.mean(value)
    result["recall"] = recall
    result["precision"] = precision
    result["miou"] = miou.mean()
    return result
        




if __name__ == "__main__":
    label_path = '../../../data/unet_image/SegmentationClass'
    predict_path = '../../../data/unet_image/predict_result'
    result = compute_indicators_dir(label_path, predict_path)
    print(result)
    # res_miou = []
    # for pred_im in os.listdir(predict_path):
    #     label = scale_image_predict(os.path.join(label_path, pred_im))
    #     pred = scale_image_predict(os.path.join(predict_path, pred_im))
    #     l, p = np.array(label).astype(int), np.array(pred).astype(int)
    #     print(set(l.reshape(-1).tolist()), set(p.reshape(-1).tolist()))
    #     miou = compute_miou(p,l,5)
    #     res_miou.append(miou)
    # print(np.array(res_miou).mean(axis=0))




    # image_predict = scale_image_predict("../../data/result/catdog.png")
    # image_label = scale_image_predict("../../data/unet_image/SegmentationClass/catdog.png")
    # p, l = np.array(image_predict), np.array(image_label)
    # p, l = p.reshape(-1), l.reshape(-1)
    # list = list(set(l.reshape(-1).tolist()))
    # out = confusion_matrix(l, p, labels=list)
    # we have got the confusion_matrix, then we will calculate
    # TP, TN, FP, FN based on the confusion_matrix.
    # because we have multi classes, so we should use for circle to calculate each class.
    
    # print(type(set(p.reshape(-1).tolist())))
    # print(set(l.reshape(-1).tolist()))
    # ious = compute_iou(p, l, 5)
    # print(ious)
    
    # test compute_indicators function.
    # segmentation_path = "../../data/unet_image/SegmentationClass/catdog.png"
    # predict_path = "../../data/result/catdog.png"
    # recall, precision, miou = compute_indicators(segmentation_path, predict_path)
    # print(recall, precision, miou)

    # 



