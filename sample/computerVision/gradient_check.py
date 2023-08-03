import numpy as np



def eval_numerical_gradient(f, x, verbose = True, h = 0.00001):
    fx = f(x)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h
        fxph = f(x)
        x[ix] = oldval - h
        fxmh = fx
        x[ix] = oldval
        grad[ix] = (fxph - fxmh) / (2 * h)
        if verbose:
            print(ix, grad[ix])
        it.iternext()
    return grad

'''
 * @Author: weiyutao
 * @Date: 2023-07-21 08:27:34
 * @Parameters: 
 * @Return: 
 * @Description: this function evaluate a numeric gradient for a function that accepts a 
 * numpy array and return a numpy array
 '''
def eval_numerical_gradient_array(f, x, df, h=1e-5):

    # notice, this a function that the first param is function parameter.
    # we want to calculate the gradient about f for x, so the shape of gradient result is
    # similar to x. df is dout what means the gradient of the current layer out. because the da/dx = da/dh * dh/dx
    # and the df or dout means da/dh at here. so this function will return
    # df*df/dx, we can initialize the grad based on x.
    grad = np.zeros_like(x)
    # create one itertor it to traverse each element in x. we will calculate the gradient based on numerical.
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        # get the index of the element that the current iterator is pointing to.
        # notice, we want to calculate the numerical gradient about df*df/dx, then, we can use the central
        # difference. just like we want to calculate da/dx = (da/df) * (df/dx), then, we simple the da/df, 
        # use df to instead it. then, the result can be expressed used df*df/dx, df/dx we can get the derivative
        # used expression, but we can also use the numerical method what means we need not to consider the function
        # f, we just need to give the the result value of f(x) and x, only need these two numerical arrays can we 
        # calculated the numerical gradient. then we can get the result expression:
        # df*df/dx = sum[df*(f(x+h) - f(x-h))] / 2h, of course, because the x is one numpy array, so we should calculate
        # the sum of array when you are on each element iteration. so it means, each element in x has its gradient.
        # and each change for the current element can not influence the other element. or the result will not be meaningful.
        ix = it.multi_index
        # iterator to the first element.
        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval
        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()

    return grad