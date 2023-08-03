'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-07-23 16:37:43
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-07-23 16:37:43
 * @Description: some basic operation for digital image processing.
***********************************************************************'''

import cv2
import numpy as np

def harris(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dest = cv2.cornerHarris(gray, 2, 3, 0.03)
    dest = cv2.dilate(dest, None)
    # set corner color
    image[dest > 0.01 * dest.max()] = [0, 0, 255]

    cv2.imshow('dest_image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sift(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)
    image = cv2.drawKeypoints(gray, kp, image)
    cv2.imshow('sift', image)
    cv2.waitKey(0)