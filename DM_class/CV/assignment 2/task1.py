from collections import defaultdict
from matplotlib import pyplot as plt # To plot images
import skimage
import numpy as np
from scipy.stats import stats
from skimage import io
from skimage import data, img_as_float
from skimage.restoration import denoise_bilateral


###################################################################
# show images
def plot_show_two_pic(one, two, title1=None, title2=None):
    # plot two images for convience and comparation
    fig, axes = plt.subplots(1, ncols=2, figsize=(15, 8))
    axes[0].imshow(one, cmap='gray')
    axes[0].set_title(title1)
    plt.axis('off')

    axes[1].imshow(two, cmap='gray')
    axes[1].set_title(title2)
    plt.axis('off')
    plt.show()


def plot_show_one_image(output, path, name):
    # show and save image
    plt.title(name)
    plt.imshow(output)
    plt.axis('off')
    plt.savefig(path)
    plt.show()
    print("image has been saved!")


###################################################################
# task1: k-means for cartoon
def update_centroids(centroids, hist):
    # update and return centroids and cluster group
    while True:
        # if key not exist, return list []
        groups = defaultdict(list)
        # assign pixel values
        for i in range(len(hist)):
            if hist[i] == 0:
                continue
            d = np.abs(centroids - i)
            index = np.argmin(d)
            groups[index].append(i)   # save the smallest number into cluster

        # find the new centroids in cluster
        new_centroids = np.array(centroids)
        for i, indice in groups.items():
            if np.sum(hist[indice]) == 0:
                continue
            new_centroids[i] = int(np.sum(indice*hist[indice])/np.sum(hist[indice]))

        # if new centroids == old centroids, find it and break
        if np.sum(new_centroids - centroids) == 0:
            break
        centroids = new_centroids
    return centroids, groups


def k_means(hist, alpha):
    # k-means algorithm, hist shape (:,1)       # p-value threshold for normaltest
    # print(alpha)
    N = 128                 # minimun group size for normaltest
    centroids = np.array([0])  # assign 0 to centroids

    while True:
        centroids, groups = update_centroids(centroids, hist)  # assign starting centroids
        # 1. start increase K if possible
        new_centroids = set()     # use set to avoid same value when seperating centroid
        for i, indice in groups.items():
            #if there are not enough values in the group, do not seperate
            if len(indice) < N:
                new_centroids.add(centroids[i])
                continue

            # judge whether need to seperate the centroid by testing if the values of the group is under a normal distribution
            z, d_alpha = stats.normaltest(hist[indice])
            if d_alpha < alpha:
                # not a normal dist, seperate
                left = 0 if i == 0 else centroids[i-1]
                right = len(hist)-1 if i == len(centroids)-1 else centroids[i+1]
                delta = right-left
                if delta >= 3:
                    c1 = (centroids[i]+left)/2
                    c2 = (centroids[i]+right)/2
                    new_centroids.add(c1)
                    new_centroids.add(c2)
                else:
                    # though it is not a normal dist, but no extra space to seperate
                    new_centroids.add(centroids[i])
            else:
                # normal dist, no need to seperate
                new_centroids.add(centroids[i])
        if len(new_centroids) == len(centroids):
            break
        else:
            centroids = np.array(sorted(new_centroids))
    return centroids


####################################################################
# cartoon methods
def CartoonNizer_1(image):
    """Source paper: Learning the k in k-means by Greg Hamerly, Charles Elkan"""
    print("###################################################################")
    print("1.1 use k-means to cartoon image")
    row, col, channels = image.shape
    #print("image shape: ", row, col, channels)

    # use histogram to get every channel datasets
    hists = []
    for i in range(channels):
        hist, _ = np.histogram(image[:, :, i], bins=np.arange(256 + 1))
        # print(hist.shape) # look the shape for kmeans
        hists.append(hist)

    # k-means method
    centroids = []   # save central points of channels
    for h in hists:
        # centroids.append(k_means(h))
        centroids.append(k_means(h, alpha=0.01))  # bigger alpha, more cartoon!
    #print("centroids:", centroids[0].shape, centroids[0], len(centroids))  # check the centroids

    # perform the method
    image = image.reshape((-1, channels))   # reshape the image for clustering
    for i in range(channels):
        channel = image[:, i]   #  all data of one channel
        # make the size of channels equal to centroids, subtract the centroids and get the new pixel
        index = np.argmin(np.abs(channel[:, np.newaxis] - centroids[i]), axis=1)
        image[:, i] = centroids[i][index]   # the same cluster, the same number
    image = image.reshape((row, col, channels))   # reshape back to get the cartoon image

    # save cartoon image into output file
    path = r"images\output\1_1_knn_cartoon.jpg"
    plot_show_one_image(image, path, name="kmeans method")
    return image


def CartoonNizer_2(image):
    print("###################################################################")
    print("1.2 use bilateral filtering to cartoon image")

    # add noisy
    # noisy = src_image + 0.6 * src_image.std() * np.random.random(src_image.shape)
    # noisy = np.clip(noisy, 0, 1)
    output = denoise_bilateral(image,sigma_color=0.01,sigma_spatial=5, multichannel=True)
    return output


def CartoonNizer_3(image):
    print("###################################################################")
    print("1.3 bilateral filtering 8 times to cartoon image")
    for i in range(8):
        print("bilateral filtering process: ", i)
        output = CartoonNizer_2(image)  # continue the part 2
        image = output

    path = r"images\output\1_3_1_bilateral_cartoon_part3.jpg"
    plot_show_one_image(output, path, name="8 times bilateral filtering")
    return output


###########################################################################
image_path = r"images\task1.jpg"  # image path
image_original = io.imread(image_path)  # original

###########################################################################
# task 1
image = io.imread(image_path)
output_1_1 = CartoonNizer_1(image)     # cartoon
#plot_show_two_pic(image_original, output_1_1, title1="orginal image", title2="k-means cartoon image")   # show them together

###########################################################################
# task 2
image_path = img_as_float(io.imread(image_path))
output_1_2 = CartoonNizer_2(image_path)
path = r"images\output\1_2_bilateral_cartoon.jpg"
plot_show_one_image(output_1_2, path, name="bilateral")  # save cartoon
#plot_show_two_pic(image_original, output_1_2, title1="orginal image", title2="bilateral_filtering cartoon image")

###########################################################################
# task 3_1
output_1_3_1 = CartoonNizer_3(image)  # already saved in function
#plot_show_two_pic(image_original, output_1_3, title1="orginal image", title2="bilateral_filtering cartoon image")

# task 3_2: kmeans + bilateral
output_1_3_2 = img_as_float(output_1_1) # output_1_1 from kmeans
output = CartoonNizer_2(output_1_3_2)   # do the bilateral
path = r"images\output\1_3_2_kmeans_bilateral_cartoon.jpg"
plot_show_one_image(output, path, name="kmeans and bilateral")  # save cartoon
# plot_show_two_pic(image_original, output_1_3_2, title1="orginal image", title2="bilateral_filtering cartoon image")