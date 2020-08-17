from matplotlib import pyplot as plt # To plot images
from skimage.color import rgb2gray # To convery RGB image to grayscale
from skimage.feature import corner_harris, corner_subpix, corner_peaks # Library for Harris corner detection
from skimage import data # To input standard images
from skimage import io
import os
import numpy as np



# file = os.listdir("picture")
file_name = "image"
# file = os.listdir(r"E:\chrome download\image")
file = os.listdir(file_name)
print(file)

color_info = {}   # save the image n-d-array
edge_number = {}  # save the mean and variance of image
add_final = 0
for i in file:
    if "task1" in i:
        continue
    #file_path = os.path.join("picture", i)
    file_path = os.path.join(file_name, i)
    print(file_path)
    input_img = io.imread(file_path)
    # handle in the same size
    # input_img = input_img[:400, :300]
    plot_shwo_one_img(input_img)
    # plt.imshow(input_img)
    # plt.show()

    GrayInputImage = rgb2gray(input_img)

    mean = np.mean(GrayInputImage)
    var = np.var(GrayInputImage)
    print(GrayInputImage.shape)

    # pad the  matrix
    # temp = np.zeros(shape=(640,480))
    # padimg = np.pad(GrayInputImage, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
    # print(padimg.shape)
    #plot_show_two_pic(input_img, GrayInputImage, title1=None, title2=None)
    color_info[i] = GrayInputImage
    add_final = np.add(GrayInputImage, add_final)
    edge_number[i] = [mean, var]

print(color_info.keys())
# print(edge_number.keys())
img_sort = sorted(edge_number.items(), key=lambda a:a[1][1])
image_name = [x[0] for x in img_sort]
print(image_name)
plt.imshow(add_final)
plt.show()
final_img = np.concatenate((color_info[image_name[0]], color_info[image_name[1]],
                color_info[image_name[2]], color_info[image_name[3]],color_info[image_name[4]]))

final_img1 = np.concatenate((color_info[image_name[0]], color_info[image_name[1]],
                color_info[image_name[2]], color_info[image_name[3]],color_info[image_name[4]]), axis=1)
# # final_img = np.append(color_info[i])
# # print(color_info[i])
# print(final_img.shape)
# # final = np.resize(final_img, new_shape=(1000, 600))
plot_show_two_pic(final_img, final_img1)