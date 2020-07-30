from skimage import data
from skimage import io
import numpy as np
from skimage.color import *
import matplotlib.pyplot as plt
from skimage import feature
import skimage


def plot_show_two_pic(one, two, title1=None, title2=None):
    # plot two images for convience
    fig, axes = plt.subplots(1, ncols=2, figsize=(15, 8))
    plt.xticks([])
    plt.yticks([])
    axes[0].imshow(one, cmap='gray')
    axes[0].set_title(title1)
    axes[1].imshow(two, cmap='gray')
    axes[1].set_title(title2)
    plt.show()

def plot_img(data):
    plt.xticks([])
    plt.yticks([])
    plt.imshow(data)
    plt.show()

def kMeans(X, K, maxIters = 10, plot_progress = None):
    # kmeans for 2 dimension
    centroids = X[np.random.choice(np.arange(len(X)), K), :]
    for i in range(maxIters):
        # find the central point
        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
        # Move centroids step
        centroids = [X[C == k].mean(axis = 0) for k in range(K)]
        if plot_progress != None:
            plot_progress(X, C, np.array(centroids))
    return np.array(centroids) , C

def CartoonNizer_ID(image):
    return image

image = io.imread("task1.jpg")
print(image.shape)
plot_img(image)

ans = []
for i in range(image.shape[2]):
    x = image[:,:, i]
    print("start processing: ",x.shape)
    centroids, C = kMeans(x, 3)
    print("Before: ",C.shape)
    C.reshape((C.shape[0],x.shape[1]))
    print(C.shape)
    ans = np.stack(C, axis=0)
    print(ans.shape)

print(ans.shape)
plot_img(ans)