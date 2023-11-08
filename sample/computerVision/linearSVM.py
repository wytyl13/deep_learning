'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-05-24 16:12:48
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-05-24 16:12:48
 * @Description: this file is dedicated to defining the function about linear svm algorithm.
 * you should distinguish the difference between nn and svm, the former has the train process.
 * and the last has not the train process. so the last has the low efficient.
 * SVM is linear model, it is one best choice for one small samples. its mathmatical logic is very graceful.
 * linear separable. nonlinear separable. if there is one straight line can distinguish between samples.
 * then, there will be countless lines can be. then, which line is the best? then, we can find the best line
 * based on the concept that which line can tolerate higher measurement error. then, how to define the line, 
 * first, we should define one standard to test the performance of each lines, and then find the best performance line.
 * then, our proble will be find one standard performance indicators.
 * then, we can give the performance standard. move one initialized line, move the line until to the first point
 * for each classes. then,  find the max distance between each moved succesfull lines. then, these lines will
 * be the best lines for the classification. but until here, you should notice there are also countless lines
 * suitable for the performance standard, but they are at least parallel to each other. what we should do is to 
 * find the center of all the lines.
 * then the max maigin is necessary for svm to find the best line. the moved succesful line is the support vector.
 * then, we should transform the max margin to one mathmatic problem.
 * max margin is equal to the math problem as follow:
 * minimize ||W||^2 and subject to yi[W^T@Xi + b] ≥ 1 (i = 1~n)
 * why? you should notice that they have the same plane for W^T@Xi + b and a*W^T@Xi + a*b, a∈R+
 * the distance from one point x0, y0 to one plane W1X+W2y+b = 0. is d=|W1x0+W2y0+b|/sqrt(W1^2+w2^2)
 * then we will extend to the generally problem. just like the set of all point what means one vector X.
 * the distance from one vector to one super plane W^T@X+b=0. is d=|W^T@X+b|/||W||, we want to maximum the d value.
 * then, if the X is the support vector what measn it is the first moved succesful line to the class.
 * then, how to maximum d value? we can scale the parameters (W, b) used a coefficient to reach one efficent that
 * the molecular of d formula |W^T@X+b| ≥ 1, this must can be implementation. then, we can get the best simple
 * formula. 
 * d = |W^T@X+b|/||W||. can simple to find the minimum ||W|| and subject to |W^T@X+b| ≥ 1.
 * only this can we get the maximum d value. so this is generally performance standard for the svm to find
 * the best line. but you should notice, |W^T@X+b| some time can be negative, because it is negative predict.
 * so we can use the formula minimize ||W||^2 and subject to yi*|W^T@X+b| ≥ 1 to handle this problem.
 * so the last performance standard to find the best line is to optimize the d.
 * what the d can transform to the conditions that should meet the condition as follow.
 * minimize ||W||^2 and subject to yi*|W^T@X+b| ≥ 1. then,  why 1, can it be other number? 2? 3? ...?
 * of course, no matter how big the number is, we can get the same plane. becaue (W, b) and (aW, ab) has
 * the same plane. the different just be the a coefficient. notice, the premise we can calculate the W and b
 * based on the performance standard above is the linear separable. or we will not calculate the W and b based on
 * the formula above.
 * linear separable formula.
 * all the linear separable should suitable for these conditions as follow.
 * 1, if yi = 1, then W^T@Xi+b ≥ 0
 * 2, if yi = -1, then W^T@Xi+b < 0.
 * in concusion, yi*[W^T@x+b]≥0, this is same as the formula we have define for svm above.
 * at last, in order to convenient to derivative, we can add the coefficient 1/2 before the ||W||^2
 * so the last performance standard for svm is minimize 1/2*||W||^2 and subject to yi*|W^T@X+b| ≥ 1
 * this performance formula can be named as quadratic programming problem, what the objective function is
 * second item and the subject condition is one item. this problem just has two result, one is there is no solution.
 * one is there is just one optimal solution. this is convex optimization problem. and the machine learning problem
 * are both the convex optimization problem. or we will not handle the problem.
 * so you can find the mathmatic concept of the svm is to transform the original problem to one convex optimization problem.
 * the convex optimization just has one global optimal solution.
 * then, we have handled the prblem that the linear classificaton used svm. then how to handle the nonlinear separable problem
 * used svm?
 * then the concept is to add one slack variable to make the formula has the solution when the samples are nonlinear separable.
 * then, we can give the formula for the nonlinear separable probles.
 * minimize 1/2*||W||^2 + CΣepsilon_i and subject to yi*|W^T@X+b| ≥ 1 - epsilon_i, epsilon_i ≥ 0. epsilon_i is the slack variable.
 * you can consider the CΣepsilon_i as the regression term. the range value of C is from -15 to 15.
 * why the regression term can handle the nonlinear separanble problem. you can image that regression term can
 * increase the dimension of the original image. then, the low dimension nonlinear separable can be linear separable
 * after adding the dimension. so this concept is this. we can increase the dimension by adding the regression
 * term to handle the nonlinear separable for low dimension.
 * so we can has the concept to handle all the classification problems.
 * no matther the problem is linear separable or nonlinear separable, we should also handle them used the 
 * linear separable to handle it.
 * what is linear separable? we can transform the performance standard to one convex optimization problem.
 * this problem has the single solution. notice, the nature of adding the regression term is also to transform
 * the problem from nonlinear to linear separable. the concept is to increase the dimension.
 * then the linear separable problem must can find the weight and b parameters to make the positive predict is
 * greater than 0 and the nagative predict is less than 0. it measn the predict value is positive if the positive 
 * predict. or the predict value is negative.
 * how to increase the dimension. just like one mapping function φ(x), x[a, b] --φ--> φ(x)[a^2, b^2, a, b, ab].
 * the two dimension is nonlinear separable, just like
 * x1[0, 0], x2[1, 1], x3[1, 0], x4[0, 1], x1 and x2 is class 1, and x3 and x4 is class 2.
 * it is nonlinear separable, you can not find the wieght and b to make the positive predict value is positive and
 * the negative predict value is negative. but you can increase the dimension by the mapping function φ(x).
 * φ(x1)[0, 0, 0, 0, 0], φ(x2)[1, 1, 1, 1, 1], φ(x3)[1, 0, 1, 0, 0], φ(x4)[0, 1, 0, 1, 0]
 * we can find the weight and b, W[-1, -1, -1, -1, 6] and b[1]
 * W.T@φ(x1) + b = 1
 * W.T@φ(x2) + b = 3
 * W.T@φ(x1) + b = -1
 * W.T@φ(x1) + b = -1
 * so you can separable these two class based on the weight and b by increasing the dimension of the dimension of samples.
 * then, how to define the φ function?
 * first, φ function is infinite dimension. because the higher dimensions the bigger probability
 * to be linear separable. the concept is we need not to known the infinite dimension mapping
 * function, we just need to find one kernel function what suitable for the expression as follow.
 * K(x1, x2) = φ(x1).T * φ(x2)
 * you can find the k function is a value. we will give some kernel function as follow.
 * 1 gaussian kernel. K(x1, x2) = e^(-(||x1-x2||^2/2σ^2))
 * 2 multi polynomial kernel. K(x1, x2) = (X1.T@X2+1)^d, d is the order number of the multi polynomial
 * 3 linear kernel, K(x, y) = x.T@y, it has the same complex and efficient like the original formula.
 * 4 tanh kernel. K(x, y) = tanh(β*x.T@y+b). tanhh(x) = [e^x-x^(-x)]/[e^x+e^(-x)]
 * these two kernel function can both be broken down into φ(x1).T@φ(x2) and subject to
 * two condition what is:
 * 1 condition, K(x1, x2) = K(x2, x1)
 * 2 condition, 存在一个常数Ci和向量Xi, Σij=1_n CiCjK(Xi, Xj) ≥ 0.
 * then, how to optimize the problem used kernel function when we just known the kernel and
 * do not known the φ?
 * 
***********************************************************************'''
import numpy as np


'''
 * @Author: weiyutao
 * @Date: 2023-05-24 17:39:06
 * @Parameters: 
 * @Return: 
 * @Description: notice, the loss function about svm algorithm is hinge loss.
 * Li = Σmax(0, sj - syi + δ) = L(y, W^T * xi + b) = max(0, 1-yi(W^T@xi+b))
 * Li is the loss value, sj is each score value we have calculated.
 * syi is the score that correct label correspond to. δ is one constant value.
 * we can define it used any number, just like we have used 1 in the follow case.
 * notice sj is each scores except the score that correct label correspond to.
 * reg can be defined used λ*ΣΣWkl^2
 * L = 1/N*ΣLi + λ*reg
 * because have the max function, so the original function hinge loss can not be differentialed.
 * so we should use subgradient.
 * dL/dW = -yixi if 1-yi(W^T@xi+b) > 0, otherwise 0
 * dL/db = -yi if 1-yi(W^T@xi+b) < 0, otherwise 0.
 * the svm dedicated to find the maximum separation hyperplane.
 * the support vector are all the vectors that smallest distance to the decision boundary.
 * the aims that decision boundary is to maximum the distance to the support vectors.
 * notice the first char is the predict result true or false, and the second char is the predict class.
 * T means the recognize result is true, and F means the recognize result is false.
 * TP, recognize true for the positive samples.
 * TN, recognize true for the negative samples.
 * FP, recognize false for the negative samples.
 * FN, recognize false for the positive samples.
 * TP+FN=1, why? TP means the number that recognized true for the positive samples.
 * FN means the numbers that recognized false for the positive samples.
 * the denominator are all the numbers of positive samples. and the sum of molecular about 
 * TP and FN is the positive samples. so it is equal to 1.
 * in the same way, FP+TN=1.
 * as the increament of TP, FP will also be increased. and as the increament of FP, TP
 * will also be increased. this is because if the number of recognized true for the positive
 * samples increased, it will result to the numbers of recogning false for the negative samples.
 * so it means we will identify more negative samples as the positive class.
 * in the same way, the increament of FP will result to the reducemnet of TN. and the reducement
 * of TN will also result to the increament of FP, because if we identify more negative samples
 * as the positive class, then the TN will be reduced. and the increament of TP will result to
 * the reducement of FN.
 * ROC curve, receiver operating characteristic curve, 接受者操作特性曲线。
 * the horizontal axis is FP, and the vertical axis is TP.
 * then which curve has the best performance, we should notice that we can get the biggest 
 * TP when we selected the smallest FP. so it means the standard we selected the 
 * best perfomance curve is the curve that biggest TP value when the FP is equal to zero.
 * but more detailes, we should consider the specific application scenario to determined
 * which curve is the best. just like the curve as follow.
 * the curve 1 has the biggest TP value than curve 2 when the FP is equal to zero.
 * but the curve 1 has the smaller TP value than curve 2 when the FP is equal to 0.5.
 * then how to select the curve? based on the specific application scenario. you should
 * select curve 1 if your applicaiton scenario is face recognization. because you want 
 * get the greatest accuracy under the condition of the smaller error value. becaue you 
 * do not want the increament of the FP value. but some other scenario, you might need
 * to get the max accuracy even the model has the bigger FP value.
 * of course, we also have some other selected standard, just like the eer. but no matter which you selected, 
 * you should known do not simple use recognition rate to judge the performance of the model.
 * because the higher recognition rate, the performance is not necessarily the best.

 * then, how to handle the multi classification problem used SVM?
 * just like the logical regression, you can judge one class with another. and for loop.
 * then, we will define the svm loss function as follow.
 * input:
        W: a numpy array of shape (D, C) containing weights, D is the feature numbers for one image.
            and C is the class numbers.
        X: a numpy array of shape (N, D) containiing a minibatch of data. N is the batchsize,  and D is the feature numbers for one image.
        y: a numpy array of shape (N,) containing training labels; is range from 0 to C.
        reg: regularization strength.
 * then, we can get the loss function of svm from above concet.
 * minimize 1/2*||W||^2 and subject to yi*|W^T@X+b| ≥ 1, we can simple this problem in order to we can 
 * handle this loss function used computer code. then, we can get the final loss function.
 * Li = Σj≠yi max(0, sj - syi + 1). syi is the score that predict true. and sj is score that each predict false.
 * just like we have predicted one cat image. and we have got the predict score list. just like we have three classes.
 * the we can image that predict value list is [3, 2, 1], the first is correspond to the cat predict value, second is car
 * and third is dog, then we can get the Li is equal to max(0, 2-3+1) + max(0, 1-3+1). then, we can get the bigger loss
 * value if we have predicted false. so we can minimize the loss function to get the best weights and intercept item.
 * then, we should calculate the average of all loss, this expression is 1/N*Σ_i=1 to N (Li), and you can find even if you
 * changed the score, it will not influence the results, because the hinge loss function just will punishment those 
 * predict false class that the score is similar to the score of predict true class. and the range value of Li is
 * from 0 to infinity. of course, you can also define the loss function as Σj≠yi max(0, sj - syi + 1)^2, they will
 * have the same efficient.
 * of course, we should consider the regularization, you can use L1 Σ|W|, L2 ΣW^2 and consider the L1 and L2 ΣβL2+L1
 * the regularization can simple the model what means the regularization can reduce the complex or the capacity of
 * the model, it can reduce the numbers of weights by add some conditions to make some weights as zero. of course, 
 * the L1 regularization prefer to make more weight value as zero, but it is not the best result. because the more zero
 * for weight value will just parts processed overfitting problems. the best method to handle the overfitting problem is
 * to simple the model and make each weights value more equal, just like the weights as follow.
 * weight1 [1, 0, 0, 0]
 * weight2[0.25, 0.25, 0.25, 0.25]
 * X[1, 1, 1, 1]
 * weight1.T @ X = weight2.T @ X = 1. they have the same score, but the weight2 can handle the overfitting model as 
 * the more efficient than weight1. this is because weight2 make each weights value more equal. then we can find
 * that regularization can simple the model and drop some unecessary noise.
 * we have defined the loss function used svm, of course, we can define it used cross-entropy loss functon
 * and other loss function.
 * this svm loss function is linear classifier, we can optimal the svm loss function to implment the nonlinear
 * classifier. softmax is one transform type,  the concept of it is similar to the logical regression.
 * they are both to transform the score as the probability. we can use the softmax function in torch. we
 * have inplemented it in the chapter1 python file at deep learning micro program directory. then softmax
 * is the premise that we can use the cross-entropy loss function. now assume we have got the probability matrix.
 * then how to define the loss value based on the softmax probability value? we can use -log(the probability of 
 * the true class). because we want to get the biggest value for the Li if the predict probability for the true
 * class has a big value. so we should log the probability value of the correct class and negative it. based on this,
 * we will not concentrate on the error predict class probability. this is maximum likelihood estimation function
 * or cross-entropy loss function. why log the probability? because the concept if likelihod estimation is 
 * to calculate the product of the probability of each samples. but because the probability is range from 0 to 1.
 * so we will get one very small value. this is not convenient to judge, so we can use log to transform the 
 * product to sum. just like p1, p2, p3, p4. these four samples probability value. we want to calculate the p1*p2*p3*p4
 * log(p1*p2*p3*p4) = log(p1) + log(p2) + log(p3) + log(p4). but because the log(0 to 1) is one negative value.
 * so we should add negative symbol before log. so it is the reason why we use the log. of course, you can use
 * the other method to handle this problem. then, we can get the loss value will be very small if the model is 
 * better. because the log(1) = 0. and the 1 is the biggest probability value. so the target is to minimize the loss
 * funtion value Σ-(log(pi)).
 * then, we have learned the hinge loss function and the cross-entropy loss function.
 * the former is svm algorithm and last is generally used for the deep learning. and they can both handle the multi
 * classification and nonlinear classification problems. and the last function should use softmax to transform
 * the score to probability first. and you should notice that softmax can increase the difference between the correct
 * score and error score.
 '''
def svm_loss_naive(W, X, y, reg):
    # initialize the gradient as zero. and the num_classes is the shape[1] of W. num_train is the shape[0] of X.
    # reg is the coefficient of regularization term.
    dW = np.zeros(W.shape)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        # scores is the probabilty value list. notice the dimension about X and W. X[i] @ W means the ith training data
        # scores for each classes.
        scores = X[i].dot(W)
        # y[i] means the label for the current sample i.
        # scores[y[i]] means the score that we have calculated for the correct label.
        # then notice, the score is not the max value. because W is begin from the initilize
        # value.
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            # except the score that correct label correspond to.
            # y[i] is the correct label.
            if j == y[i]:
                continue
            # scores[i] is each score except the score that correct label correspond to.
            # i is the δ value.
            margin = scores[j] - correct_class_score + 1

            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]
    # loss is the sum value, we should calculate the average.
    loss /= num_train
    # add the regularization to the loss
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += reg * 2 * W
    return loss, dW


'''
 * @Author: weiyutao
 * @Date: 2023-05-24 16:15:14
 * @Parameters: 
 * @Return: 
 * @Description: this function used vectorized method. it is more efficient than
 * using for loop to implement it.
 '''
def svm_loss_vectorized(W, X, y, reg):
    loss = 0.0

    # initialize the gradient as zero.
    # dw has the same shape as W, the dimension is (D, C) and the dimension of X is (N, D)
    dW = np.zeros(W.shape)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    # scores = X @ W = (N, D) @ (D, C) = (N, C)
    scores = np.dot(X, W)
    # get the correspond score list based on the label y. and reshape the result to (num_train, 1)
    correct_class_score = scores[np.arange(num_train), y].reshape(num_train, 1)
    # calculate the margin based on the np.maximum function.
    margin = np.maximum(0, scores - correct_class_score + 1)
    # because we have calculated the Li about the score of predict true class and itself. so we should reset the value.
    # as zero. because its value is 1. it will influence the final loss value.
    # find the correspond correct lable index in each row and reset the value as 0.
    margin[np.arange(num_train), y] = 0
    # then calculate the average of the margin array and consider the regularization.
    # notice we should divide into the num_train, this number is equal to the numbers
    # of classes reduction of 1.
    loss = np.sum(margin) / num_train + reg * np.sum(W * W)
    # if the element in margin is greater than 0, it means the score that false classes is
    # greater than the score of correct class.
    # because the hinge loss function can not be derivatived, we can calculate the 
    # partial derivative used the subgradient. so it means we can do it as follow.
    # hinge loss function  L = max(0, score - correct_score + 1) = max(0, 1 - (correct_score - score))
    # so it means if correct_score - score < 1, the loss value is non zero.
    # if correct_Score - score >= 1, the loss value is zero.
    # then, we can filter the correspond score. if margin = loss value, if margin > 0, it means
    # the correspond loss has the partial derivative, we can calculate the partial derivative
    # of score - correct_score + 1.
    # then, we will implement the hinge loss function used another method.
    # just like the predict score we have got based on the W.T @ X + b = y^.
    # then how to define the hinge loss function?
    # we are handling the binary classification problem. just like the true label is 1 or -1.
    # then, we can define the hinge loss function as follow.
    # L(xi, yi) = max(0, 1 - y_i * (X @ W)), we can get loss value is zero if we have predicted true.
    # of course the predict true involved predict true for positive and for the negative.
    # if predict true for positive, max(0, 1 - 1 * 1) = 0, if predict true for negative.
    # max(0, 1 - -1 * -1) = 0. so we can get the zero value if we have predicted true.
    # then, we can go on to next code. how to calculate the dw?
    # dw = σL/σw = 1/N * (1/2*||W||^2]' + [1 - y_i * (X @ W)]') + (reg*W^2)' = 1/N * (||W|| - y_i*X) + 2*reg*W.
    # notice, reg is the regularization coefficient.
    # notice, margin > 0 mean the correspond score is the error classification.
    # so then, how to define the gradient of the hinge loss function?
    # in this function, our goal is to minimize the loss function. so in this case scenery. we hope
    # the different score value is greater than one between correct class and incorrect class. why?
    # because hinge loss function L = max(0, 1 - (correct_score - incorrect_score)), if 
    # correct_score - incorrect_score >= 1, the loss value is zero. this is simple to understand, then, how to 
    # define the gradient? for optimize this problem, we should calculate the gradient that the hinge loss function
    # for the weight W.
    # dw = σL/σw = 1/N * (1/2*||W||^2]' + [1 - y_i * (X @ W)]') + (reg*W^2)' 
    # = 1/N * (||W|| - y_i*X) + 2*reg*W.
    
    # we will just consider those samples that contributes to the loss value.
    # then, we will implement the gradinet for the hinge loss function. we should use margin varible
    # as the gradient figure. we can use it to increase the correct class score and reduce the error class score.
    # how to define the margin? we have defined margin based on the correct zero, error 1. then, we just need to
    # make the correct score in margin is negative. so we can margin[np.arange(num_train), y] -= correct_number.
    # then np.dot(X.T, margin) / num_train + reg * 2 * W.
    margin[margin > 0] = 1
    error_number = np.sum(margin, axis=1)
    margin[np.arange(num_train), y] -= error_number
    dW = np.dot(X.T, margin) / num_train + reg * 2 * W

    return loss, dW


    