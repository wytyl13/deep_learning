import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(image):
    if image.ndim == 2:
        plt.imshow(image, cmap = 'gray')
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
    plt.show()

# iterative method to calculate the threshold value.
def iterative_value(image):
    T = image.mean()
    while True:
        # THE average gray value of the background.
        T0 = image[image < T].mean()
        # the average gray value of the prospects.
        T1 = image[image >= T].mean()
        t = (T0 + T1) / 2
        if abs(t - T) < 1:
            break
        T = int(t)
    return T

#OTSU method to calculate the threshold value to segementation.
def OTSU(image):
    sigma = -1
    T = 0
    for t in range(0, 256):
        background = image[image <= t]
        prospects = image[image > t]
        # calculate the proportion about the background size / image size.
        p0 = background.size / image.size
        p1 = prospects.size / image.size

        m0 = background.mean()
        m1 = 0 if prospects.size == 0 else prospects.mean()
        sigmoid = p0 * p1 * (m0 - m1)**2

        if sigmoid > sigma:
            sigma = sigmoid
            T = t
    return T

# the adaptive threshold. we have learned the global threshold.
# then we will learn how to implement the local threshold.
# compare the current pixel value and the mean or gaussian weighted of
# its surrounding area, if the pixel is greater, set it as 255, or set it as 0.
def adaptive_threshold(image, type = 0):
    # notice, there are two method to binary the picture used adaptive_threshold.
    # one is the mean, one is the gaussion weighted value.
    # of course, we can also define the super param to enhance the effection.
    # just like if we compare the T and the pixel value. the effection
    # is not always good. so we should define the super param c.
    # notice the super param C can do better for the shaded part in the picture.
    image_binary = 0
    if type == 0:
        C = 6
        winsize = 21
        # calculate the mean of the winsize*winsize range of the current pixel.
        image_blur = cv2.blur(image, (winsize, winsize))
        # if image > image_blur 255, else 0
        image_binary = np.uint8(image > (image_blur.astype(np.int) - C)) * 255
    else:
        # alpha is the experience value.
        # this method has the better efficient.
        alpha = 0.15
        winsize = 21
        image_blur = cv2.GaussianBlur(image, (winsize, winsize), 5)
        image_binary = np.uint8(image > (1 - alpha) * image_blur) * 255 
    show(image_binary)


if __name__ == "__main__":
    # histogram threshold. it is suitable for the bimodal histogram.
    # image = cv2.imread('c:/users/80521/desktop/bird.png', 0)
    # # plt.hist(image.ravel(), 256, [0, 256])
    # _, image_binary = cv2.threshold(image, 125, 255, cv2.THRESH_BINARY)
    # show(image_binary)


    # then, learned the trigonometry threshold. it is suitable for
    # the unimodal histogram.
    # image = cv2.imread('c:/users/80521/desktop/flower.png', 0)
    # # plt.hist(image.ravel(), 256, [0, 256])
    # th, image_binary = cv2.threshold(image, 0, 255, cv2.THRESH_TRIANGLE)
    # print(th)
    # show(np.hstack([image, image_binary]))

    # image = cv2.imread('c:/users/80521/desktop/bird.png', 0)
    # T = iterative_value(image)
    # print(f"best threshold is equal to {T}")

    # the otsu threshold.    
    # image = cv2.imread('c:/users/80521/desktop/bird.png', 0)
    # threshold_value, image_binary = cv2.threshold(image, -1, 255, cv2.THRESH_OTSU)
    # print(threshold_value)

    # print(OTSU(image))
    image = cv2.imread('c:/users/80521/desktop/word.png', 0)
    # image_binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                      cv2.THRESH_BINARY, 21, 8)
    # show(image_binary)
    adaptive_threshold(image, 1)