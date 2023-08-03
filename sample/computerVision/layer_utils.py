'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-07-18 09:16:58
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-07-18 09:16:58
 * @Description: this file is dedicated to defining the layer utils. just like the
 * forward and backward.
 * notice, we should distingush the size of all parameters.
 * X(m, n_x), W1(n_x, n_h), Z1(m, n_h), W2(n_h, c)
***********************************************************************'''
import numpy as np




def affine_forward(X, W, b):
    out = None
    # because the input data shape is (m, (3, 32, 32)), so we should reshape
    # the input data shape first.
    X_vector = X.reshape(X.shape[0], -1)
    # we should consider the forward expression.
    # X_vector(m, n_x), W1(n_x, n_h), X_vector @ W1 = (m, n_h), you should notice
    # the dimension is different from the above we have considered.
    out = np.dot(X_vector, W) + b
    # this cache variable stored all the parameters we have used current.
    cache = (X, W, b)
    return out, cache


def relu_forward(Z1):
    # this function is dedicated to nonlinear mapping.
    # we will use relu mapping function and we should return the param Z1 and the 
    # vlaue we have got used relu algorithm.
    out = None
    out = np.maximum(0, Z1)
    cache = Z1
    return out, Z1


'''
 * @Author: weiyutao
 * @Date: 2023-07-18 11:10:39
 * @Parameters: dout: the derivation passed from the former layer.
 * @Return: dx: gradient with respect to x. notice the derivation of relu.
 * d relu(Z1)/d(Z1), if Z1 > 0. the result is equal to Z1. or the result is equal to dZ1/d(Z1) = 1.
 * @Description: 
 '''
def relu_backward(dout, cache):
    # notice the partial derivatives of relu function is as above.
    # if the value is greater than zero, the partial derivatives value is 1. 
    # and if the value is less than zero, the partial derivatives value is zero.
    # relu is one function mapping. and the max(0, X @ W1 + b1) functipn gradient with
    # respect to x is equal to 0, if X @ W1 + b1 < 0, W1 if X @ W1 + b1 > 0.
    # then, we can get the result as follow.
    # we can get the x what is the param 
    dx, x = None, cache
    # dx = Z1, if Z1 <= 0, the derivation is zero.
    dx = x
    dx[dx < 0] = 0
    dx[dx > 0] = 1
    dx *= dout

    return dx




'''
 * @Author: weiyutao
 * @Date: 2023-07-18 09:18:19
 * @Parameters: 
 * @Return: 
 * @Description: this function involved affine_forward and relu_forward what means
 * W1 @ X + b1 = Z1 and A1 = relu(Z1)
 * notice the difference between out and dout, the out variable is the result we calculated based on
 * the forward function, and dout variable is the gradient of the out layer what involved 
 * all the layer that behind the current layer. so you should notice this problem.
 * you should notice the difference between forward and backward function, no matter relu, sigmoid and other 
 * nonlinear function for the layer, this featuer is similar.
 '''
def affine_relu_forward(X, W1, b1):
    Z1, W_cache = affine_forward(X, W1, b1)
    A1, relu_cache = relu_forward(Z1)
    cache = (W_cache, relu_cache)
    return A1, cache

'''
 * @Author: weiyutao
 * @Date: 2023-07-18 12:16:02
 * @Parameters: cache: the cache what we have got based on the affine_forward function.
 * dout: the derivation passed from the former layer. the dimension of dA is (m, n_h)
 * @Return: dX = dZ / dx = d(X@W+b)/dX = W. dW = X. db = 1.
 * then, you should consider the derivation that passed from the former layer.
 * and you should consider the size of the derivation result.
 * @Description: 
 '''
def affine_backward(dA, cache):
    X, W, b = cache
    dX, dW, db = None, None, None
    # dA(m, n_h), W(n_x, n_h)
    # dX(m, n_x) is equal to X(m, n_x)
    dX = np.dot(dA, W.T).reshape(X.shape)
    X_vector = X.reshape(X.shape[0], -1)
    # dW(n_x, n_h) is equal to W. (n_x, m) @ (m, n_h) = (n_x, n_h)
    dW = np.dot(X_vector.T, dA)
    # db(n_h, 1) is equal to (n_h, m) @ (m, 1) = (n_h, 1)
    db = np.dot(dA.T, np.ones(X.shape[0]))

    return dX, dW, db


'''
 * @Author: weiyutao
 * @Date: 2023-07-18 12:06:05
 * @Parameters: dZ, the derivation what passed from the former layer.
 * @Return: dx what the derivation of relu function respect to the Z1.
 * just like if we want to calculate the derivation of relu, we should do as follow.
 * max(0, W1 @ X + b1)' = [0 + d(max(0, W1 @ X + 1))/ d(W1 @ X + 1)] * [d(W1 @ X + b1) / dX]
 * d(W1 @ X + 1)] * [d(W1 @ X + b1) / dX] is the derivation passed from the former layer.
 * what means the dout param in this function.
 * the dimension of dA is equal to A what is equal to the relu value what shape is 
 * equal to Z(m, n_h).
 * @Description: 
 '''
def affine_relu_backward(dout, cache):
    W_cache, relu_cache = cache
    # the dimension of dA is (m, n_h)
    dA = relu_backward(dout, relu_cache)
    dX, dW, db = affine_backward(dA, W_cache)
    return dX, dW, db

# notice the param X is not the training data, it is the score array.
# its shape is (num_train, num_classes).
def svm_loss(scores, y):
    loss, dx = None, None
    num_train = scores.shape[0]
    num_classes = scores.shape[1]
    correct_class_score = scores[np.arange(num_train), y].reshape(num_train, 1)
    # notice the difference between np.max and np.maximum function. the last generally used for
    # the array. notice the meaningful of hinge loss function. it is used for the svm classification.
    # its function is to measure the difference between the correct class and error class.
    # if the difference value is greater than 1, we will not do the punishment. and you should notice
    # the difference have some condition what means max(0, 1 - (correct_score - incorrect_score))
    # if correct_score - incorrect_score is greater than or equal 1, the loss is zero. 
    # if the result is less than 1, the loss will be greater than zero. notice the order of substracting.
    margin = np.maximum(0, scores - correct_class_score + 1)
    # notice, we can not consider the correct score element. so we should set it used zero.
    margin[np.arange(num_train), y] = 0
    # calculate the average.
    loss = np.sum(margin) / num_train
    # then, we will calculate the derivative, if the value is less than or equal zero. 
    # the derivative value is zero. or it will be 1.
    margin[margin > 0] = 1
    correct_number = np.sum(margin, axis=1)
    margin[np.arange(num_train), y] -= correct_number
    dx = margin / num_train

    return loss, dx




def softmax_loss(X, y):
    loss, dX = None, None
    num_train = X.shape[0]
    scores = X - np.max(X, axis = 1).reshape(num_train, 1)
    normalized_scores = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(num_train, 1)
    loss = -np.sum(np.log(normalized_scores[np.arange(num_train), y]))
    loss /= num_train

    normalized_scores[np.arange(num_train), y] -= 1
    dX = normalized_scores / num_train

    return loss, dX
