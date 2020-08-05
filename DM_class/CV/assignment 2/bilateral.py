import numpy as np
import random
from skimage import io,color
from matplotlib import pyplot as plt


def show_image(image, name):
    plt.imshow(image)
    plt.title(name)
    plt.show()


def image_3d_2d(image, channel):
    row, col, _ = image.shape
    output = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            output[i][j] += image[i][j][channel]
    return output


def color_gaussian(gsigma, winsize):
    # gaussian for color, 3d
    r = int(winsize/2)
    c = r
    kernel = np.zeros((winsize, winsize))
    sigma1 = 2 * gsigma * gsigma
    for i in range(-r, r+1):
        for j in range(-c, c+1):
            # the formula
                kernel[i + r][j + c]= np.exp(-float(float(i*i + j*j))/sigma1)
    return kernel


def space_gaussian(image, ssigma):
    # gaussian for space
    output = np.zeros(image.shape, np.uint8)
    thres = (1 - ssigma) * 255  # threshold

    # for 2 d image, need to split rgb image channel
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            r = random.randint(0,255)
            if r < thres*255:
                output[row][col] = 0
            elif r > 230:  # adjust
                output[row][col]= 200
            else:
                output[row][col] = image[row][col]
    return output


def bilateral_filter(image, gsigma, winsize, ssigma):
    r = int(winsize/2)
    c = r
    # image1 = np.pad(image, ((r, c),(r, c)), constant_values=0)
    # #image1 = sp_noise(image1, prob=0.01)
    # image = image1
    row, col = image.shape
    sigma2 = 2*ssigma*ssigma
    gkernel = color_gaussian(winsize, gsigma)  # color gaussian
    kernel = np.zeros((winsize, winsize))  # space gaussian

    bilater_image = np.zeros((row, col))  # save bilater output

    for i in range(1, row - r):
        for j in range(1, col - c):
            for z in range(1, channel):
                skernel = np.zeros((winsize, winsize))
                # print(i, j)
                for m in range(-r, r + 1):
                    for n in range(-c, c + 1):
                        # define the formule
                        skernel[m + r][n + c] = np.exp(-pow((image[i][j] - image[i + m][j + n]), 2) / sigma2)
                        # kernel = space * color
                        kernel[m + r][n + c] = skernel[m + r][n + r] * gkernel[m + r][n + r]
                sum_kernel = sum(sum(kernel))
                kernel = kernel / sum_kernel
                for m in range(-r, r + 1):
                    for n in range(-c, c + 1):
                        bilater_image[i][j] = image[i + m][j + n] * kernel[m + r][n + c] + bilater_image[i][j]

    return bilater_image


print("""g""")
file_path = r"images\task1.jpg"
image = io.imread(file_path)
row, col, channel = image.shape
show_image(image, "original")
print(image.shape)
output = np.zeros_like(image.shape)

image_2d = image_3d_2d(image, 0)  # 3d image into 2d
show_image(image_2d, "image_2d")
bilater_image = bilateral_filter(image_2d, gsigma=3, winsize=3, ssigma=3)

for c in range(1, channel):
    image_2d = image_3d_2d(image, c)  # 3d image into 2d
    show_image(image_2d, "image_2d")
    bilater_image_tmp = bilateral_filter(image_2d, gsigma=3, winsize=3, ssigma=5)
    #bilater_image = np.concatenate((bilater_image, bilater_image_tmp), axis=0)
    bilater_image = np.dstack((bilater_image, bilater_image_tmp))
    print(bilater_image.shape)
    #show_image(bilater_image, "bilater_image")

print(bilater_image.shape)
show_image(bilater_image, "bilater_image")
# image_gray = color.rgb2gray(image)
# show_image(image_gray)



