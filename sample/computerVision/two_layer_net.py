'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-07-20 08:44:54
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-07-20 08:44:54
 * @Description: this file, we will define the two layer net neural network
 * used original python language, and you should notice, these class or function
 * is not necessary for our working, we just need to how to define them, we should
 * use pytorch or tensorflow to define the model. not it.
***********************************************************************'''
from full_connected_net import TwoLayerNet
from data_utils import get_CIFAI10_data
from layer_utils import affine_forward
from data_utils import rel_error
from layer_utils import affine_backward
from gradient_check import eval_numerical_gradient_array
from layer_utils import relu_forward
from layer_utils import relu_backward
from layer_utils import affine_relu_forward
from layer_utils import affine_relu_backward
from layer_utils import svm_loss
from layer_utils import softmax_loss
from gradient_check import eval_numerical_gradient


import numpy as np

if __name__ == "__main__":
    # data = get_CIFAI10_data()
    
    """ 
    for k, v in list(data.items()):
        print(('%s: ' %k, v.shape)) 
    """
    # then, we will test the forwar function, you should notice that
    # we want to learn one weight variable that is suitable for all the samples.
    # so the size of weight is not related to the sample numbers. it just related to
    # the input image size and the output size. just like the number of neurous in hidden
    # layer. so we can define these test samples size based on the concept above.
    # notice the function np.prod. 
    """     
    num_inputs = 2
    input_shape = (4, 5, 6)
    output_dim = 3
    input_size = num_inputs * np.prod(input_shape)
    weight_size = output_dim * np.prod(input_shape)

    # notice, this defining suitable for the expression Z = X @ W + b
    # X(m, n_x), W(n_x, n_h), b(n_h), Z(m, n_h)
    X = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
    W = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
    b = np.linspace(-0.3, 0.1, num=output_dim)

    # notice the out variable in this case is Z(m, n_h) = (2, 3)
    out, _ = affine_forward(X, W, b)
    correct_out = np.array([[1.49834967, 1.70660132, 1.91485297],
                            [3.25553199, 3.5141327, 3.77273342]])
    print('difference: ', rel_error(out, correct_out))     
    """              

    # test the affine layer backward
    """     
    np.random.seed(231)
    X = np.random.randn(10, 2, 3)
    W = np.random.randn(6, 5)
    b = np.random.randn(5)
    # notice, the dout varible in this case is Z
    dout = np.random.randn(10, 5)

    _, cache = affine_forward(X, W, b)
    dX, dW, db = affine_backward(dout, cache) 
    """

    """
    dictionary = {'dX': dX, 'dW': dW, 'db': db}
    print(dictionary) 
    """

    # then we can calculate the gradient based on the number itself.
    # we can define one gradient calculation function to get the result, and compare
    # the result with the gradient dX, dW, db that calculated based on the affine_backward
    # function. notice the recall rule for the function that has the function parameter.
    # you should use lambda expression to pass the function param.
    """
    dX_numerical = eval_numerical_gradient_array(lambda x: affine_forward(X, W, b)[0], X, dout)
    dW_numerical = eval_numerical_gradient_array(lambda x: affine_forward(X, W, b)[0], W, dout)
    db_numerical = eval_numerical_gradient_array(lambda x: affine_forward(X, W, b)[0], b, dout)

    print("dX error: ", rel_error(dX_numerical, dX))
    print("dW error: ", rel_error(dW_numerical, dW))
    print("db error: ", rel_error(db_numerical, db)) 
    """

    # test the relu function
    """ 
    # relu forward function testing.
    X = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)
    out, _ = relu_forward(X)
    correct_out = np.array([[0., 0., 0., 0.], 
                            [0., 0., 0.04545455, 0.13636364],
                            [0.22727273, 0.31818182, 0.40909091, 0.5]])
    print('difference: ', rel_error(out, correct_out))

    # relu backward function.
    np.random.seed(231)
    X = np.random.randn(10, 10)
    dout = np.random.randn(*X.shape)

    dX_numerical = eval_numerical_gradient_array(lambda x: relu_forward(X)[0], X, dout)

    _, cache = relu_forward(X)
    dX = relu_backward(dout, cache)
    print("difference: ", rel_error(dX_numerical, dX)) 
    """

    # test affine_relu_forward and affine_relu_backward function.
    # notice the difference between  out and dout variable 
    """     
    np.random.seed(231)
    X = np.random.randn(2, 3, 4)
    W = np.random.randn(12, 10)
    b = np.random.randn(10)
    dout = np.random.randn(2, 10)

    out, cache = affine_relu_forward(X, W, b)
    dX, dW, db = affine_relu_backward(dout, cache)

    dX_numerical = eval_numerical_gradient_array(lambda x: affine_relu_forward(X, W, b)[0], X, dout)
    dW_numerical = eval_numerical_gradient_array(lambda x: affine_relu_forward(X, W, b)[0], W, dout)
    db_numerical = eval_numerical_gradient_array(lambda x: affine_relu_forward(X, W, b)[0], b, dout)

    print("dX error: ", rel_error(dX_numerical, dX))
    print("dW error: ", rel_error(dW_numerical, dW))
    print("db error: ", rel_error(db_numerical, db)) 
    """

    """     
    np.random.seed(231)
    num_classes, num_inputs = 10, 50
    X = 0.001 * np.random.randn(num_inputs, num_classes)
    y = np.random.randint(num_classes, size=num_inputs)
    dX_numerical = eval_numerical_gradient(lambda x: svm_loss(X, y)[0], X, verbose = False)
    loss, dX = svm_loss(X, y)
    print("svm_loss: ", loss)
    print("svm difference: ", rel_error(dX_numerical, dX))

    dX_numerical = eval_numerical_gradient(lambda x: softmax_loss(X, y)[0], X, verbose=False)
    loss, dX = softmax_loss(X, y)
    print("softmax loss: ", loss)
    print("softmax difference", rel_error(dX_numerical, dX)) 
    """

    # then, we will define the two layer neural network.


    """     
    np.random.seed(231)
    N, D, H, C = 3, 5, 50, 7
    X = np.random.randn(N, D)
    y = np.random.randint(C, size = N)
    std = 1e-3
    # define the instancement of class TwoLayerNet.
    model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std)
    W1_std = abs(model.params['W1'].std() - std)
    b1 = model.params['b1']
    W2_std = abs(model.params['W2'].std() - std)
    b2 = model.params['b2']
    # you should make sure that the biases of each layer at here are all equal to zero. 
    # or we should stop the process.
    assert W1_std < std / 10, 'first layer weight do not seem right'
    assert np.all(b1 == 0), 'first layer biases do not seem right'
    assert W2_std < std / 10, 'second layer weight do not seem right'
    assert np.all(b2 == 0), 'second layer biases do not seem right'

    # we should initialize the weight. X(N, D), Z1(N, H), W(D, H), b1(H), Z2(N, C), W2(H, C), b2(C)
    model.params['W1'] = np.linspace(-0.7, 0.3, num=D*H).reshape(D, H)
    model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
    model.params['W2'] = np.linspace(-0.3, 0.4, num=H*C).reshape(H, C)
    model.params['b2'] = np.linspace(-0.9, 0.1, num=C)
    X = np.linspace(-5.5, 4.5, num=D*N).reshape(D, N).T

    # test the loss function, we just calculate the scores array.
    # scores = model.loss(X)
    # correct_scores = np.asarray(
    #     [[11.53165108,  12.2917344,   13.05181771,  13.81190102,  14.57198434, 15.33206765,  16.09215096],
    #     [12.05769098,  12.74614105,  13.43459113,  14.1230412,   14.81149128, 15.49994135,  16.18839143],
    #     [12.58373087,  13.20054771,  13.81736455,  14.43418138,  15.05099822, 15.66781506,  16.2846319 ]])
    # scores_diff = np.abs(scores - correct_scores).sum()

    # print(scores_diff)
    # assert scores_diff < 1e-6, 'Problem with test-time forward pass'

    # then, we will give the label y.
    # pass the y param into the function loss. we will get both the loss and grads variabel.
    y = np.asarray([0, 5, 1])
    loss, grads = model.loss(X, y)
    correct_loss = 3.4702243556
    assert abs(loss - correct_loss) < 1e-10, 'Problem with training-time loss'
    # model.reg = 1.0
    # loss, grad = model.loss(X, y)
    # correct_loss = 26.5948426952
    # assert abs(loss - correct_loss) < 1e-10, 'Problem with regularization loss'
    for reg in [0.0, 0.7]:
        print('Running numeric gradient check with reg = ', reg)
        model.reg = reg
        loss, grads = model.loss(X, y)

        for name in sorted(grads):
            f = lambda _: model.loss(X, y)[0]
            grad_num = eval_numerical_gradient(f, model.params[name], verbose=False)
            print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name]))) 
            
    """

    input_size = 32 * 32 * 3
    hidden_size = 50
    num_classes = 10
    model = TwoLayerNet(input_size, hidden_size, num_classes)
    solver = None
    solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    lr_decay=0.95,
                    num_epochs=10, batch_size=100,
                    print_every=100, verbose = True)
    solver.train()
    print('Validation accuracy: ', solver.best_val_acc)


