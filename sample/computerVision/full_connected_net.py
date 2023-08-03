'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-07-14 22:50:46
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-07-14 22:50:46
 * @Description: this file is dedicated to defining the full connected neural network.
***********************************************************************'''
from builtins import range
from builtins import object
import numpy as np

from layer_utils import affine_relu_forward
from layer_utils import affine_forward
from layer_utils import softmax_loss
from layer_utils import affine_backward
from layer_utils import affine_relu_backward

'''
 * @Author: weiyutao
 * @Date: 2023-07-17 17:34:14
 * @Parameters: 
 * @Return: 
 * @Description: a connected network with Relu nonlinearity and softmax loss
 * that uses a modulan layer design. we assume an input dimension of D, a hidden
 * dimension of H, and perform classification over C classes. the architecture should
 * be affine --> relu --> affine --> softmax.
 * notice, we will not define the gradient descent in this class.
 * and you should notice the learned parameters of the model are stored in the dictionary
 * self,.params that maps parameter names to numpy arrays.
 * then, we should notice the important content of this TwoLayerNet class.
 * this neural network has two layer nueral network, one hidden layer and one output layer.
 * the input shape is (n_x, m) what n_h means the neurons numbers of the hidden layer. and the m means the sample numbers.
 * the hidden layer shape is (n_h, m) and the W1 parameters shape is (n_h, n_x), and the b1 shape is (n_h, 1),
 * so the Z1 = W1 @ X + b1 = (n_h, m). the output layer shape is (c, m). so the W2 shape is (c, n_h), so
 * Z2 = W2 @ tanh(Z1) + b2 = (c, m). so we can conclude the rule that the input shape is equal to (n_h, m). you should notice
 * that m dimension is the vertical axis. and  the horizontal axis is feature numbers n_h. and you should notice the rule
 * that the shape of all the weights, it is equal to (the first dimension of the next layer, the first dimension of former layer)
 * just like the shape of W1 param, we can find its shape based on the last hidden layer and the former input layer.
 * the last layer shape for the W1 is (n_h, m), and the former input layer shape for W1 is (n_x, m), then, we can conclude
 * that the shape of W1 is equal to (n_h, n_x). so we can define the TwoLayerNet class based on the former concept
 '''
class TwoLayerNet(object):
    def __init__(
        self, 
        input_dim = 3 * 32 * 32,
        hidden_dim = 100, 
        num_classes = 10, 
        weight_scale = 1e-3,
        reg = 0.0
    ) -> None:
        self.params = {}
        self.reg = reg
        # initialize the weights and biases of the two layer net.
        # you should notice the size of weights. the size of input set
        # is (m, n), m is the sample numbers and n is the feature numbers for one image.
        # X @ w = (m, n) @ (n, the numbers of hidden layer neural unit what means hidden_dim in this class)
        # weight_scale is the scale number. 0 is the mean of the normal distribution.
        W1 = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.params['W1'] = W1
        b1 = np.zeros(hidden_dim)
        self.params['b1'] = b1
        W2 = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['W2'] = W2
        b2 = np.zeros(num_classes)
        self.params['b2'] = b2


    def loss(self, X, y=None):

        scores = None
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        A1, relu_cache = affine_relu_forward(X, W1, b1)
        scores, cache = affine_forward(A1, W2, b2)

        if y is None:
            return scores
        
        loss, grads = 0, {}
        loss, d_scores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.sum(W1 * W1) + 0.5 * self.reg * np.sum(W2 * W2)
        dX, grads['W2'], grads['b2'] = affine_backward(d_scores, cache)
        dX, grads['W1'], grads['b1'] = affine_relu_backward(dX, relu_cache)
        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2

        return loss, grads