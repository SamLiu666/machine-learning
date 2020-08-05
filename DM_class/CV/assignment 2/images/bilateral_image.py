import numpy as np
import random, time
from skimage import io,color
from matplotlib import pyplot as plt


def distance(i, j, k, l):
    # (i,j) - (k,l)
    return np.sqrt((i-k)**2 + (j-l)**2)

def gaussian(x, sigma):
    p1 = 1.0 / 2 * np.pi * (sigma**2)
    p2 = np.exp(-(x**2)/(2*sigma**2))
    return p1 * p2


def apply_bilateral_filter(image_source, filtered_image, x, y, winsize, sigma_color, sigma_sapce):
    hl = int(winsize / 2)
    denominator = 0
    molecule = 0
    i = 0
    while i < winsize:
        j = 0
        while j < winsize:
            neighbour_x = x - (hl - i)
            neighbour_y = y - (hl - j)

            if neighbour_x >= len(image_source): neighbour_x -= len(image_source)
            if neighbour_y >= len(image_source[0]): neighbour_y -= len(image_source[0])

            # print(image_source[neighbour_x][neighbour_y])   # check the number
            g_color = gaussian(image_source[neighbour_x][neighbour_y] - image_source[x][y], sigma_color)
            g_space = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_sapce)

            w_temp = g_color * g_space
            denominator += image_source[neighbour_x][neighbour_y] * w_temp
            molecule += w_temp

            j += 1
        i += 1

    denominator = denominator / molecule

    # denominator = denominator.reshape(-1, 1)
    # print(denominator[0], type(denominator), denominator.shape)
    # ans = int((denominator[0]))
    # print(denominator[0], type(denominator), denominator.shape, ans)
    # filtered_image[x][y] = ans
    filtered_image[x][y] = int(denominator)
    # filtered_image[x][y] = int(np.round(denominator))


def bilateral_filter_own(image, filter_diameter, sigma_color, sigma_sapce):
    # 2 d
    filtered_image = np.zeros(image.shape)

    i = 0
    while i < len(image):
    # while i < 10:
        # if i%100 == 0:
        #     print("############################")
        #     print(i, "times")
        #     time.sleep(2)
        j = 0
        while j < len(image[0]):
        # while j < 10:
            apply_bilateral_filter(image, filtered_image, i, j, filter_diameter, sigma_color, sigma_sapce)
            j += 1
        i += 1
    return filtered_image


def image_3d_2d(image, channel):
    row, col, _ = image.shape
    output = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            output[i][j] += image[i][j][channel]
    return output


start = time.time()
image = io.imread("task1.jpg")
row, col, channels = image.shape
print(row, col, channels)
plt.imshow(image)
plt.show()

# image1 = image_3d_2d(image, 0)
# print(image1.shape)
# plt.imshow(image1)
# plt.show()
# output = bilateral_filter_own(image, filter_diameter=5, sigma_color=12.0, sigma_sapce=16.0)

output = np.zeros((row, col, channels))

for channel in range(channels):
    start = time.time()

    image_temp = image_3d_2d(image, channel)  # turn into 2-d image for bilateral_filter_own
    # plt.imshow(image_temp)
    # plt.show()
    # biggger sigma_color bigger cluster, bigger sigma_sapce more points
    filtered_image = bilateral_filter_own(image_temp, filter_diameter=10, sigma_color=20.0, sigma_sapce=50.0)
    # plt.imshow(filtered_image)
    # plt.show()

    for i in range(row):
        for j in range(col):
            print("###########")
            print(filtered_image[i][j])

            output[i][j][channel] = int(filtered_image[i][j])  # save

            print(output[i][j][channel])

    end = time.time()
    print("time cost: ", end - start)

    time.sleep(2)

plt.imshow(output)
plt.show()
end = time.time()
print("time cost: ", end-start)