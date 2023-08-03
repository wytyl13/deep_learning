#!C:/Users/80521/AppData/Local/Programs/Python/Python38 python
# -*- coding=utf8 -*-

'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-05-24 13:16:36
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-05-24 13:16:36
 * @Description: this file, we will define the  lineara classifier class what based on 
 the gradient descent function. notice it is different from the knn algorithm.
 it will use the derivation to make the minimize the loss function. the knn algorithm
 is to find the min k L1 or L2 distance.
 *
 * then, let us consider the feature project what means we should not just use the original
 * feature in image to as the input data. we can extract the important feature. just like the
 * histogram feature, feature project is important for the machine learning. then the feature
 * extract from the original image is trained by the convolusion nueral network in deep learning.
 * the network will train the convolusion kernel to find the feature from the original image.
 * then,  let's consider the deeper problems, whether we can on the basis of original data
 * to extract the characteristics of the image, and then based on the extracted feature image
 * to train the weights. just like the balanced histogram feature and so on, the difference is 
 * this method need to define the feature method by ourselves, but the deep learning method
 * need not us to define the feature extracted method, we can train all the model based on
 * to add the convariance into the first step of the model. so we can implement to auto learning
 * the convariance kernel in deep learning. so this concept is to consider the more meaningful
 * features in one image. not consider each pixel value in one image. we can consider two, three
 * four and so on pixel values as one feature. just like the face, we can consider the eye as the training
 * features. not the pixel value of eye region in one image.
 * then, we have learned two method to implement the nonlinear classification. one is add the dimensions
 * of the features image. one is change the classifier to make it to be the nonlinear classifier. the last method
 * we can use the deep learning what means the created the neural network to implement it.
 * the generally sigmoid function,  sigmoid function and relu function
 * sigmoid function: 1/(1+e^(-x))
 * relu function: max(0, W.T@X_b)
 * notice the role of sigmoid function. the sigmoid function are generally the nonlinear.
 * so it can make our model to handle the nonlinear problems. just like if we dropped the sigmoid function
 * in the deep learning model. the model can be expressed used linear expression no matter how much layers
 * you model have. so the nonlinear sigmoid function is meaningful to make you model can handle the nonlinear
 * problems.
 * sigmoid function: make the result value to 0-1
 * tanh function, make the result value to -1~1
 * relu function, max(0, x), make the result value to 0~infinity
 * leaky relu function, max(0.1*x, x), make the result value to 0.1x~x
 * we generaly used the relu function.
 * one standard neural network invovled two layer at least. input, hidden and output layer.
 * the layers of one model is equal to the sum of hidden layer and output layer.
 * if we want to handle the binary classification problems. we can define the output layer used one neurons.
 * if we want to handle n classification problems. we can define the output layer usde n neurons. each neurons connect
 * to the former layer and net layer, so we can named it as all connection neural network. 
 * then, the neural network can have many hidden layer, so it can make our model more efficient for the nonlinear problems.
 * but it can also make our model overfitting. but if you have bigger samples as your training dataset, you will
 * not meet this problems.

***********************************************************************'''
import numpy as np
from linearSVM import svm_loss_vectorized
from linearSVM import svm_loss_naive
from softmax import softmax_loss_vectorized

import time
import random
import numpy as np
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt
import math
import pdb

# set the default value for the plt, set the size of the figure, the element value
# and the color of the image.
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

class LinearClassifier(object):
    def __init__(self) -> None:
        self.W = None
    

    '''
     * @Author: weiyutao
     * @Date: 2023-05-24 14:08:19
     * @Parameters: 
            X: a numpy array of shape (N, D) containing train data; N samples and D features.
            y: a numpy array of shape (N, ) containing training labels;
            learning_rate: the learning rate for optimizing.
            reg: regularization strength.
            num_iters: number of steps to take when optimizing.
            batch_size: number of training examples to use at each step.
            verbose: boolean, if true, print progress during optimization.
     * @Return: a list containing the value of the loss function at each training iteration. 
     * @Description: you should input the original data and this function will return a list
     that stored the loss value for each iteration.
     notice, the batchsize in this function means to select part data from the train dataset. just like
     the train data number is 49000, but we just selected 200 as the train data. and iteration numbers is 1500.
     we have tested the result about the different batch_size, they are similar but the bigger batch_size wasted
     more times.
     '''
    def train(
        self, 
        X, 
        y, 
        learning_rate = 1e-3, 
        reg = 1e-5, 
        num_iters = 100, 
        batch_size = 200, 
        verbose = False
    ):
        num_train, dim = X.shape
        num_classes = np.max(y) + 1

        # init the original weight for the instancement linearClassifier.
        # how to calculate the dimension of W? 
        # it is based on the input data X, the dimension of X is (N, D)
        # then X @ W = (N, D) @ (D, num_classes) = (N, num_classes)
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)


        # define the loss varibale.
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            # then, how to define the data for each batch. we have define the batch_Size
            # so we can use the choice function in numpy to do it.
            # notice replacement is true is more efficient than false.
            # this function will return one list indicies that number of batch size, 
            # we have stored it used mask variable.
            mask = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[mask]
            y_batch = y[mask]

            # we will define the loss function at last.
            # the different method will have different loss function. 
            # notice, the grad is the derivation of weight, so it has the same shape as W.
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # iterate the weight param based on the learning rate and grad what the foremer
            # learning rate is the drop step length and the last grad is the drop slope.
            # because we are falling, so you should use the negative of slope.
            self.W -= learning_rate * grad
            if verbose and it % 100 == 0:
                    print("iterator %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history
    


    '''
     * @Author: weiyutao
     * @Date: 2023-05-24 14:38:27
     * @Parameters: 
     * @Return: 
     * @Description: this function will predict the data used the trained successful weight.
     '''
    def predict(self, X):
        # init one container to store the predict value.
        # it should has the same as the y_label
        y_pred = np.zeros(X.shape[0])
        # notice, the dimension of X @ W is (N, num_classes), 
        # the row numbers is N, and the number of columns is num_classese.
        # each value for each row is the probability value for the classes index.
        # so you should find the index of the max probabilty value in the result.
        # and it is the predict class. stored them used y_pred we have defined
        # in the former code.
        y_pred = np.argmax(np.dot(X, self.W), axis=1)

        return y_pred




    '''
     * @Author: weiyutao
     * @Date: 2023-05-24 14:32:08
     * @Parameters: 
            X_batch: a numpy array of shape (batch_size, D) containing a minibatch
            sample data of N.
            y_batch: a numpy array of shape (batch_size, ) containing a minibatch 
            y labels of N.
            reg: regularization strength.
     * @Return: a tuple containing the loss value and gradient with respect to self.W
        what is an array of the same shape as W.
     * @Description: this function is dedicated to calculate the loss and derivative.
     '''
    def loss(self, X_batch, y_batch, reg):
        # but we have not define any code in this function.
        # because it is inside of the parent class LinearClassifier.
        # we will define the implement of this loss function
        # in other child class.
        pass


class LinearSVM(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)



class Softmax(LinearClassifier()):
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)


if __name__ == "__main__":
    #  test the linear classification used svm.
    print("WHOAMI")
    cifar10_dir = 'data/cifar-10-batches-py'
    # clear the previouse variable we will define as follow in the memory.
    try:
        del X_train, y_train
        del X_text, y_test
        print('Clear previously load data.')
    except:
        pass

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    # test print 1
    """ 
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)
    """

    # imshow the figure. 7*10, test plot the figure.
    """  
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()
    """

    # the cross validation. the dimension of X_train dataset is 50000, 32, 32, 3.
    # the dimension of test data is 10000, 32, 32, 3.
    # the numbers of samples is 50000. we defined the numbers of cross validation is 1000.
    # then, we can select part of the data from the original data.
    num_training = 49000
    num_validation = 1000
    num_test = 1000
    num_dev = 500

    # we will define the training dataset and the validation dataset based on these defining above.
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    # we will also make a development set, which is a small subset of the training set.
    # notice, the param replace means random selected the data that don't repeat.
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    # define the part dataset of the test variable.
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # test the selected training, validation and test data set.
    """  
    print('train data shape', X_train.shape)
    print('train labels shape', y_train.shape)
    print('validation data shape', X_val.shape)
    print('validation labels shape', y_val.shape)
    print('test data shape', X_test.shape)
    print('test labels shape', y_test.shape)
    """
    # the each input data should be the vector. so we should reshape each samples from 1, 32, 32, 3 to 1, 3072
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
    # test the data set after reshaping.
    """  
    print('training data shape', X_train.shape)
    print('validation data shape', X_val.shape)
    print('test data shape', X_test.shape)
    print('development data shape', X_dev.shape)
    """

    # Preprocessing, substract the mean image what can be also named as the decentralized.
    # notice, we should calculate each mean value for each feature element. so we can get the shape of
    # mean_image is (3072, 0), and we can imshow the mean_image
    mean_image = np.mean(X_train, axis=0)
    # imshow the mean_image
    """ 
    plt.figure(figsize=(4, 4))
    plt.imshow(mean_image.reshape((32, 32, 3)).astype('uint8'))
    plt.show() 
    """

    # then, substract the mean_image from train data and test data. notice, each variable should 
    # substact the mean_image what is calculated based on the X_train data.
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image

    # then, append the bias dimension used one for each data set.
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
    # print(X_train.shape, X_val.shape, X_test.shape, X_dev.shape)


    # then, util here we have finished all the preprocessing work, then, we can call the function about svm
    # we have defined succesful.
    # we should init the weight first, and you should notice all the weights should be very small.
    # notice the shape of weights 3073 * 10. X_train(49000, 3073)
    W = np.random.randn(3073, 10) * 0.0001
    # the second and third param is the data, becaus the X_train is very big, so we used X_dev to instead it.
    # to test the loss function. the fourth param is the coefficient of regularization.
    loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.000005)
    # print('loss: %f' % (loss, ))
    # notice, this is one cycle.

    # then we will recycle the traning process.
    """     
    svm = LinearSVM()
    tic = time.time()
    loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=2.5e4, 
                        num_iters=1500, verbose=True)
    toc = time.time() 
    """
    # print('that took %fs' % (toc - tic))
    # plot the loss value. plot the loss_history.
    """     
    plt.plot(loss_hist)
    plt.xlabel('Iteration number')
    plt.ylabel('loss value')
    plt.show() 
    """
    # test the predict accuracy for X_train and x_val data set. X_train must have more
    # accuracy than x_val.
    """ 
    y_train_pred = svm.predict(X_train)
    y_val_pred = svm.predict(X_val)
    print('training accuracy: %f' %(np.mean(y_train == y_train_pred)))
    print('validation accuracy: %f' %(np.mean(y_val == y_val_pred)))
    """

    # test the rate and regularization list, and get all the results of accuracy.
    # and selected the best accuracy value. of course, only based on the val dataset, 
    # the accuracy is meaningful.
    results = {}
    best_val = -1
    # store the best svm model parameters.
    best_svm = None
    learning_rate = [1e-7, 1e-6]
    regularization_strength = [2.5e4, 5e4]
    for rate in learning_rate:
        for regularization in regularization_strength:
            svm = LinearSVM()
            loss = svm.train(X_train, y_train, learning_rate = rate, reg=regularization, num_iters=1500, verbose=False)
            y_train_pred = svm.predict(X_train)
            training_accuracy = np.mean(y_train == y_train_pred)
            y_val_pred = svm.predict(X_val)
            validation_accuracy = np.mean(y_val == y_val_pred)
            if validation_accuracy > best_val:
                best_val = validation_accuracy
                best_svm = svm
            results[(rate, regularization)] = (training_accuracy, validation_accuracy)
    
    # print the results.
    """ 
    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy))
    print('best validation accuracy achieved during cross-validation: %f' % best_val)  
    """

    # visualize the cross-validation results
    # log the each value in results. the x is the learning rate and the y scatter is regularization strength.
    # and you should notice the accuracy value. we will use the color to distingush the different accuracy.
    """     
    x_scatter = [math.log10(x[0]) for x in results]
    y_scatter = [math.log10(x[1]) for x in results]

    # plot training accuracy
    marker_size = 100
    colors = [results[x][0] for x in results]
    # define the figure 2*1 and located at the first image.
    plt.subplot(2, 1, 1)
    # set the padding for the subplot
    plt.tight_layout(pad=3)
    # the last param cmap is the color mapping.
    plt.scatter(x_scatter, y_scatter, marker_size, c=colors, cmap=plt.cm.coolwarm)
    # show the colorbar
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('CIFAR-10 training accuracy')

    # plot validation accuracy. it is just the accuracy value different from the training accuracy.
    colors = [results[x][1] for x in results]
    plt.subplot(2, 1, 2)
    plt.scatter(x_scatter, y_scatter, marker_size, c=colors, cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('CIFAR-10 validation accuracy')
    plt.show() 
    """

    # test the best_svm model we have trained.
    """ 
    y_test_pred = best_svm.predict(X_test)
    test_accuracy = np.mean(y_test_pred == y_test)
    print('linear SVM on raw pixels final test set accuracy: %f' % test_accuracy) 
    """


    # imshow the weights we have trained, but you should notice strip out the bias.
    w = best_svm.W[:-1, :]
    w = w.reshape(32, 32, 3, 10)
    w_min, w_max = np.min(w), np.max(w)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        # rescale the weights to range from 0 to 255. or we will not inshow the weight as the image successfully.
        # .squeeze() is dedicated to reshape one weight from (32, 32, 3, 1) to (32, 32, 3)
        # rescale processing, 255.0 * (orginal - min) / (max - min)
        wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])
    plt.show()







