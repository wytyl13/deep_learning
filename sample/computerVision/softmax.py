'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-07-09 20:03:17
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-07-09 20:03:17
 * @Description: of course, we should define the program about softmax what can transform the
 score value to the probability.
***********************************************************************'''
from builtins import range
import numpy as np
from random import shuffle


def softmax_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)

    num_train = X.shape[0]
    scores = np.dot(X, W)
    scores -= np.max(scores, axis=1).reshape(num_train, 1)
    normalized_scores = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(num_train, 1)
    loss = - np.sum(np.log(normalized_scores[np.arange(num_train), y]))
    loss /= num_train
    loss += reg * np.sum(W * W)
    normalized_scores[np.arange(num_train), y] -= 1
    dW = np.dot(X.T, normalized_scores)
    dW /= num_train
    dW += reg * 2 * W

    return loss, dW


