import numpy as np
from sklearn.datasets import load_sample_image
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

def plot_img(image):
    plt.imshow(image)
    plt.axis("off")
    plt.show()

china = load_sample_image("china.jpg")[80:360, 70:390]
flower = load_sample_image("flower.jpg")[80:360, 130:450]
dataset = np.array([china, flower], dtype=np.float32)
print(dataset.shape)
batch_size, height, width, channels = dataset.shape

# 2 filters
filter_test = np.zeros(shape=(7,7, channels, 2), dtype=np.float32)
filter_test[:, 3, :, 0] = 1 # vertical
filter_test[3, :, :, 1] = 1 # horizontal

plot_img(filter_test[:,:,0,0])
plot_img(filter_test[:,:,0,1])

#######################################################################
# use tf.nn.conv2d
output = tf.nn.conv2d(batch_size, filter_test, strides=[1,2,2,1], padding="SAME")
print("Output shape:" + str(output.shape))
plt.imshow(output[0, :, :, 0], cmap='gray') # plot 1st image's 1nd feature map, channel 0
plt.show()
plt.imshow(output[0, :, :, 1], cmap='gray') # plot 1st image's 2nd feature map, channel 1
plt.show()

plt.imshow(output[1, :, :, 0], cmap='gray') # plot 2nd image's 1nd feature map, channel 0
plt.show()
plt.imshow(output[1, :, :, 1], cmap='gray') # plot 2nd image's 2nd feature map, channel 1
plt.show()