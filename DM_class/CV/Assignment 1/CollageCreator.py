from matplotlib import pyplot as plt # To plot images
from skimage.color import rgb2gray # To convery RGB image to grayscale
from skimage.feature import corner_harris, corner_subpix, corner_peaks # Library for Harris corner detection
from skimage import data # To input standard images
from skimage import io
import os
import numpy as np


def plot_show_two_pic(one, two, title1=None, title2=None):
    # plot two images for convience
    fig, axes = plt.subplots(1, ncols=2, figsize=(15, 8))
    plt.axis('off')
    axes[0].imshow(one, cmap='gray')
    axes[0].set_title(title1)

    axes[1].imshow(two, cmap='gray')
    axes[1].set_title(title2)
    plt.axis('off')
    plt.show()


def plot_show_one_img(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def CollageCreate(AddressofFolder):
    """
    :param
    AddressofFolder: a relative address of folder containing images from a video
    :return and displays a collage image
    """


    # 1. find the image  directory and create data structure
    file = os.listdir(AddressofFolder)
    #print(file)    # check the image path

    image_info = {}  # save the input_gray_image
    image_variance = {}  # save the mean and variance of image
    add_directly = 0    #  for directly add the picture

    # 2. find all the image and save
    for i in file:
        # except the task1 image
        if "task1" in i:
            continue

        file_path = os.path.join(AddressofFolder, i)
        print(file_path)   # check the image path
        input_img = io.imread(file_path)
        plot_show_one_img(input_img)   # output the  origianl image

        GrayInputImage = rgb2gray(input_img)
        # print(GrayInputImage.shape)  # check image shape, should be the same

        variance = np.var(GrayInputImage)  # choose the variance parameter
        add_directly = np.add(GrayInputImage, add_directly)

        image_info[i] = GrayInputImage
        image_variance[i] = variance

    # plot_show_two_pic(input_img, add_final, title1="original image", title2="directly add image")
    # plot_show_one_img(add_directly)
    print("--------------------------------------------\n")
    img_sort = sorted(image_variance.items(), key=lambda a: a[1])  # sort in decreasing variance
    # save the sorted image name
    # print(image_variance)
    image_name = [x[0] for x in img_sort]
    print("before sorted : ",image_info.keys(), "\n", "after sorted : ",image_name)

    # collage the image
    img_out_0 = image_info[image_name[0]]
    img_out_1 = image_info[image_name[0]]
    for i in image_name[1:]:
        img_out_0 = np.concatenate((img_out_0, image_info[i]), axis=0)
        img_out_1 = np.concatenate((img_out_1, image_info[i]), axis=1)
    plot_show_two_pic(img_out_0, img_out_1, title2="collage image horizontally", title1="collage image vertically")
    return img_out_0, img_out_1


if __name__ == '__main__':
    AddressofFolder = "image"
    (img_out_0, img_out_1) = CollageCreate(AddressofFolder)
    print(img_out_0)