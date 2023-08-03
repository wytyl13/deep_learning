'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-05-07 19:15:59
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-05-07 19:15:59
 * @Description: this file is to load the datasets CIRFI10 for the classifier application.
***********************************************************************'''
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import platform

def load_pickle(f):
    version = platform.python_version_tuple()
    if(version[0] == '2'):
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding="latin1")
    raise ValueError("invalid python version: {}".format(version))



def load_CIFAR_batch(fileName):
    """ load one batch data, there are five batches, one batch has 10000 samples """
    with open(fileName, "rb") as f:
        datadict = load_pickle(f)
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all samples """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, "data_batch_%d" %(b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtrain = np.concatenate(xs)
    Ytrain = np.concatenate(ys)

    del X, Y
    Xtest, Ytest = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))
    print(type(Xtrain), type(Ytrain), type(Xtest), type(Ytest))
    return Xtrain, Ytrain, Xtest, Ytest

# this function is dedicated to getting all the samples directly. just like the function load_CIFAR10 function, 
# this function will return the training samples, validation samples and test samples. of course, we should pass
# some params into this function. of couse, we should use the load_CIFAR10 function to get the original data first.
def get_CIFAI10_data(
    num_training = 49000, 
    num_validation = 1000, 
    num_test = 1000,
    substract_mean = True
):
    # load the original data used the load_CIFAR10 function. and you should
    # get the data path first.
    cifar10_dir = 'data/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    # then, we can sunsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    print(X_train.shape)
    print(y_train.shape)
    # then, normalize the data. substract the mean image.
    # notice the meaning of the mean_image what mean the mean for
    # every sample for each pixel. just like the sample numbers is 49000, and the pixel 
    # numbers is 32*32*3=3072, so we can get the mean_image shape is (3072, ),
    # notice the X_train shape is (49000, 32, 32, 3), so we can subsract the mean_image
    # for the X_train samples directly.
    if substract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image
    # we should transpose the samples from 49000, 32, 32, 3 to 49000, 3, 32, 32.
    # why, this is suitable for the convolusion operation.
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # package data into a dictionary.
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }

'''
 * @Author: weiyutao
 * @Date: 2023-07-20 12:36:20
 * @Parameters: 
 * @Return: 
 * @Description: this function is dedicate to calculating the relative error.
 * maximum function will compare the first param and the second param.
 * and return the max value. we should use it to compare two matrix.
 '''
def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def imshowMatrix(dists):
    plt.imshow(dists, interpolation='none')
    plt.show()


