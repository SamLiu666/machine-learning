from skimage import data
from skimage import io
import numpy as np
from skimage.color import *
import matplotlib.pyplot as plt
from skimage.color import rgb2gray,rgb2hsv
"""https://scikit-image.org/"""


def task_1_1():
    # open the image from the dataset
    input_Image = data.coins()
    plt.imshow(input_Image)
    plt.imshow(input_Image, cmap='gray')
    plt.show()


def task_1_2():
    # open the image from system
    input_Image = io.imread('1.jpg')
    # plt.imshow(input_Image)
    plt.imshow(input_Image, cmap='gray')
    plt.show()

def task_2():
    input_Image = io.imread('1.jpg')  # ndarray
    width, height, depth = input_Image.shape
    print(width, height, depth)

    output_image = np.zeros((width, height, depth))
    print(input_Image)

    for iRow in range(width):
        # height//2 , half iamge
        for iCol in range(height//2+1):
            for iChannel in range(depth):
                output_image[iRow, iCol, iChannel] = input_Image[iRow, iCol, iChannel]

    # output_image = input_Image.copy()
    plt.imshow(output_image.astype('uint8'))
    print(output_image)
    #plt.imshow(output_image)
    plt.show()


def pixelOperatorMultiplyDivide(ival, ifactor):
    # complete the contrast changing
    iTemp =  ival // ifactor
    if iTemp > 255:
        return 255
    elif iTemp <0:
        return 0
    return iTemp


def pixelOperatorAddSub(ival, ifactor):
    # complete the contrast changing
    iTemp = int(ival + ifactor)
    if iTemp > 255:
        return 255
    elif iTemp <0:
        return 0
    return iTemp


def applyPointOperations(image, method, ifactor):
    width, height = image.shape
    for i in range(width):
        for j in range(height):
            if method == 'contrast':
                # call pixelOperatorMultiplyDivide
                image[i][j] = pixelOperatorMultiplyDivide(image[i][j], ifactor)
            if method == 'brightness':
                # call pixelOperatorAddSub
                image[i][j] = pixelOperatorAddSub(image[i][j], ifactor)

    return image


def task_3():
    input_image = data.astronaut()
    # gray: 0-1
    gray_image = rgb2gray(input_image)
    ifactor = 100

    output_image = applyPointOperations(gray_image*255, 'contrast', ifactor)
    #output_image = applyPointOperations(gray_image * 255, 'brightness', ifactor)
    # present image
    fig, axes = plt.subplots(1, ncols=2, figsize=(8,4))
    ax = axes.ravel()
    ax[0].imshow(output_image, cmap='gray')
    ax[1].imshow(gray_image, cmap='gray')
    plt.show()

def task_4(gray_value=0):
    input_image = io.imread('1.jpg')
    if gray_value == 0:
        output_image = rgb2gray(input_image)
    elif gray_value == 1:
        output_image = rgb2hsv(input_image)

    plt.imshow(output_image)
    plt.show()

def task_5_compare():
    input_image = io.imread('ironman.jpg')
    output_image1 = rgb2gray(input_image)
    output_image2 = rgb2hsv(input_image)
    fig, axes = plt.subplots(1, ncols=2, figsize=(8,4))
    ax = axes.ravel()
    ax[0].imshow(output_image1)
    ax[1].imshow(output_image2)
    plt.show()
    plt.imshow(output_image1)
    plt.axis('off')
    plt.savefig('iron1.jpg')
    plt.show()

    plt.imshow(output_image2)
    plt.axis('off')
    plt.savefig('iron2.jpg')
    plt.show()
if __name__ == '__main__':

    # task_1_1()
    # task_1_2()
    # task_2()
    # task_3()
    # task_4(1)
    task_5_compare()