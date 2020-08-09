import numpy as np
import random, time
from skimage import io,color
from matplotlib import pyplot as plt


def distance(x, y, z, k, l, m):
    # (i,j) - (k,l)
    return np.sqrt((x - k)**2 + (y - l)**2 + (z-m)**2)

def gaussian(x, sigma):
    p1 = 1.0 / np.sqrt(2 * np.pi * (sigma**2))
    p2 = np.exp(-(x**2)/(2*sigma**2))
    return p1 * p2


def bilateral_filter(image_source, x, y, z, winsize, sigma_color, sigma_sapce):
    hl = int(winsize / 2)
    denominator = 0
    molecule = 0
    i = 0
    ans = 0

    while i < winsize:
        j = 0
        while j < winsize:
            # 3 d image processing
            neighbour_x = x - (hl - i)
            neighbour_y = y - (hl - j)
            neighbour_z = z

            # when neighbour is out of range
            if neighbour_x >= len(image_source): neighbour_x -= len(image_source)
            if neighbour_y >= len(image_source[0]): neighbour_y -= len(image_source[0])

            # print(image_source[neighbour_x][neighbour_y])   # check the number
            g_color = gaussian(image_source[neighbour_x][neighbour_y][neighbour_z] - image_source[x][y][z], sigma_color)
            g_space = gaussian(distance(neighbour_x, neighbour_y, neighbour_z, x, y, z), sigma_sapce)

            w_temp = g_color * g_space
            denominator += image_source[neighbour_x][neighbour_y][neighbour_z] * w_temp
            molecule += w_temp

            j += 1
        i += 1

    denominator = denominator / molecule   # value after filtering
    # turn into rgb
    if ans>255:     ans=1.0
    elif ans<0:     ans=0.0
    else:           ans = denominator/255
    #print(ans)
    return ans


def my_own_bilateral(image):
    """:arg
    image: image source ndarray
    return: bilateral filtered image ndarray
    """
    print("1_2_2. my_onw_bilateral")
    start = time.time()

    row, col, channels = image.shape
    #print(row, col, channels)
    # plt.imshow(image)   # check the image
    # plt.show()
    # create output image
    output = np.zeros((row, col, channels))
    # assign the value to output
    for i in range(row):
        if i%100==0:
            print("counts: ", i)
        for j in range(col):
            for ch in range(channels):
                # output[i][j][ch] = bilateral_filter(image, x=i,y=j,z=ch, winsize=4, sigma_color=4, sigma_sapce=10)
                # output[i][j][ch] = bilateral_filter(image, x=i, y=j, z=ch, winsize=4, sigma_color=8, sigma_sapce=20)
                output[i][j][ch] = bilateral_filter(image, x=i, y=j, z=ch, winsize=4, sigma_color=2, sigma_sapce=10)
    end = time.time()
    print("my onw bilateral filtering time cost: ", end-start)
    return output


if __name__ == '__main__':
    image_path = r"images\task1.jpg"  # image path
    image = io.imread(image_path)
    output = my_own_bilateral(image)
    plt.imshow(output)
    plt.show()