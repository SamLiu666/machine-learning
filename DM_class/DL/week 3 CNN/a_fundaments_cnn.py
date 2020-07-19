"""Acknowledgment: this tutorial was developed based non the Chapter 14 materials from
the book Hands-on Machine Learning with Scikit-learn and Tensorflow (TF 2.x edition)"""

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
# To plot pretty figures
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
# Where to save the figures
PROJECT_ROOT_DIR = "." #the current directory
DIR_NAME = "Fundamentals"
#check for folder existence, otherwise creating a new folder
path = os.path.join(PROJECT_ROOT_DIR,  DIR_NAME)
if(not os.path.exists(path)):
    os.mkdir(path)

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, DIR_NAME, fig_id + ".png")
    print("Saving figure: ", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format="png", dpi=300)

def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")
    plt.show()

def plot_actual_image(image):
    plt.rcParams["figure.figsize"] = (image.shape[0]/50.0, image.shape[1]/50.0)
    plt.imshow(image.astype(np.uint8),interpolation="nearest", shape=[image.shape[0], image.shape[1]])
    plt.axis("off")
    plt.show()

def plot_color_image(image):
    plt.imshow(image.astype(np.uint8),interpolation="nearest")
    plt.axis("off")
    plt.show()

##########################################################################
## 1.2L oad two sample images using sklearn
from sklearn.datasets import load_sample_image

china = load_sample_image("china.jpg")
flower = load_sample_image("flower.jpg")
plot_image(china)
plot_image(flower)

##########################################################################
## 2L Convolution Layer
"""":tf.nn.conv2d(input, filter, strides, padding, name).
input is a 4-D tensor of shape  [𝑏𝑎𝑡𝑐ℎ,𝑖𝑛_ℎ𝑒𝑖𝑔ℎ𝑡,𝑖𝑛_𝑤𝑖𝑑𝑡ℎ,𝑖𝑛_𝑐ℎ𝑎𝑛𝑛𝑒𝑙𝑠] .
filter is a tensor of shape  [𝑓𝑖𝑙𝑡𝑒𝑟_ℎ𝑒𝑖𝑔ℎ𝑡,𝑓𝑖𝑙𝑡𝑒𝑟_𝑤𝑖𝑑𝑡ℎ,𝑖𝑛_𝑐ℎ𝑎𝑛𝑛𝑒𝑙𝑠,𝑜𝑢𝑡_𝑐ℎ𝑎𝑛𝑛𝑒𝑙𝑠] .
"""
import numpy as np
china = load_sample_image("china.jpg")[80:360, 70:390]
flower = load_sample_image("flower.jpg")[80:360, 130:450]
batch = np.array([china, flower], dtype=np.int)
# plot_image(china)
# plot_image(flower)
print("batch.shape:  ",  batch.shape)

print("-------------------------")
batch_size, height, width, channels = batch.shape
# Create 2 filters with  7x7x3x2
filters = np.zeros(shape=(7,7, channels, 2), dtype=np.int)
filters[:, 3, :, 0] = 1  # vertical line, why?
filters[3, :, :, 1] = 1  # horizontal line, why?
# plot_image(filters[:, 3, :, 0])
# plot_image(filters[3, :, :, 0])

import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

output = tf.nn.conv2d(batch, filters, strides=[1,2,2,1], padding="SAME")
print("-------------------------")
def my_run():
    print("Output shape:" + str(output.shape))
    fig, axes = plt.subplots(1, ncols=2, figsize=(8,4))
    ax = axes.ravel()
    ax[0].imshow(output[0, :, :, 0], cmap='gray') # plot 1st image's 1nd feature map, channel 0
    ax[1].imshow(output[0, :, :, 1], cmap='gray') # plot 1st image's 2nd feature map, channel 1
    ax[2].imshow(output[1, :, :, 0], cmap='gray') # plot 2nd image's 1nd feature map, channel 0
    ax[3].imshow(output[1, :, :, 1], cmap='gray') # plot 2nd image's 2nd feature map, channel 1
    plt.show()

output = tf.nn.conv2d(batch, filters, strides=[1,2,2,1], padding="SAME")
print("Output shape:" + str(output.shape))
plt.imshow(output[0, :, :, 0], cmap='gray') # plot 1st image's 1nd feature map, channel 0
plt.show()
plt.imshow(output[0, :, :, 1], cmap='gray') # plot 1st image's 2nd feature map, channel 1
plt.show()

plt.imshow(output[1, :, :, 0], cmap='gray') # plot 2nd image's 1nd feature map, channel 0
plt.show()
plt.imshow(output[1, :, :, 1], cmap='gray') # plot 2nd image's 2nd feature map, channel 1
plt.show()