from skimage import data
from skimage import io
import numpy as np
from skimage.color import *
import matplotlib.pyplot as plt
from skimage import feature
import skimage
from skimage.feature import corner_harris, corner_subpix, corner_peaks # Library for Harris corner detection
from scipy import ndimage
from skimage.color import rgb2gray,rgb2hsv
from scipy.ndimage.filters import convolve


# 1.Smooth image with a Gaussian filter
def gaussian(k, sigma=1):
    k = int(k) // 2
    x, y = np.mgrid[-k:k + 1, -k:k + 1]  # return 2 same indexed grid
    normal = 1 /(2.0 * np.pi * sigma**2)
    gau =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return gau


# 2.Compute magnitude and orientation of gradient
def sobel_filters(gau_value):
    # build sobel kernels and calculate Ix, Iy for the following
    x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    Ix = ndimage.filters.convolve(gau_value, x)
    Iy = ndimage.filters.convolve(gau_value, y)

    # magnitude G and the slope of the gradient
    gradient = np.hypot(Ix, Iy)
    gradient = gradient / gradient.max() * 255
    theta = np.arctan2(Iy, Ix)
    return (gradient, theta)


# 3  Compute Non-maximum suppression
def comput_Non_maximum_suppression(gradient, theta):
    # 3.1 create a zero matrix as the same size of image
    M, N = gradient.shape
    Z = np.zeros((M, N), dtype=np.int32)
    # 3.2 compute the angle
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180
    # 3.3 Check if the pixel in the same direction has a higher intensity than the pixel that is currently processed;
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255
                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = gradient[i, j + 1]
                    r = gradient[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = gradient[i + 1, j - 1]
                    r = gradient[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = gradient[i + 1, j]
                    r = gradient[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = gradient[i - 1, j - 1]
                    r = gradient[i + 1, j + 1]
                if (gradient[i, j] >= q) and (gradient[i, j] >= r):
                    Z[i, j] = gradient[i, j]
                else:
                    Z[i, j] = 0
            except IndexError as e:
                pass
    return Z

# 4.1 Define two thresholds: low and high
def threshold(image, lowThresholdRatio=0.01, highThresholdRatio=0.1):
    # define the highThreshold and  lowThreshold
    highThreshold = image.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    M, N = image.shape
    re_image = np.zeros((M, N), dtype=np.int32)
    Low_Threshold = np.int32(25)
    High_Threshold = np.int32(255)

    strong_i, strong_j = np.where(image >= highThreshold)
    weak_i, weak_j = np.where((image <= highThreshold) & (image >= lowThreshold))
    re_image[strong_i, strong_j] = High_Threshold
    re_image[weak_i, weak_j] = Low_Threshold

    return (re_image, Low_Threshold, High_Threshold)

# 4.2 complete the hysteresis
def  myCannyEdgeDetector(image, Low_Threshold, High_Threshold):
    M, N = image.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (image[i, j] == Low_Threshold):
                try:
                    if ((image[i + 1, j - 1] == High_Threshold) or (image[i + 1, j] == High_Threshold)
                            or (image[i + 1, j + 1] == High_Threshold)
                        or (image[i, j - 1] == High_Threshold) or (image[i, j + 1] == High_Threshold)
                        or (image[i - 1, j - 1] == High_Threshold) or (image[i - 1, j] == High_Threshold) or (image[i - 1, j + 1] == High_Threshold)):
                        image[i, j] = High_Threshold
                    else:
                        image[i, j] = 0
                except IndexError as e:
                    pass
    return image

def plot_show_two_pic(one, two, title1=None, title2=None):
    # plot two images for convience
    fig, axes = plt.subplots(1, ncols=2, figsize=(15, 8))
    axes[0].imshow(one, cmap='gray')
    axes[0].set_title(title1)
    axes[1].imshow(two, cmap='gray')
    axes[1].set_title(title2)
    plt.show()


######################################################################
# read picture
image_file_path = r'picture\task1.jpg'
inputImg_0 = io.imread(image_file_path)  # ndarray
inputImg = rgb2gray(inputImg_0)  # turn into gray picture

##################################################################
#
print("(i) output of skimage.feature.canny for the input image")
print("--------------------------------------------\n")
part1 = "i) skimage.feature.canny sigma=1 of input image"
part2 = "i) skimage.feature.canny sigma=3 of input image"
feature_canny_1 = feature.canny(inputImg, sigma=1)
feature_canny_2 = feature.canny(inputImg, sigma=3) # different sigma value
plot_show_two_pic(feature_canny_1, feature_canny_2, title1=part1, title2=part2)


##################################################################
# (ii) edge output of myCannyEdgeDetector();
print("--------------------------------------------\n")
print("(ii) edge output of myCannyEdgeDetector()")
img_gaussian = gaussian(k=5, sigma=1)     # 1. Gaussian  with kenel=5x5
img_smoothed = convolve(inputImg, img_gaussian)  #  Gaussian smooth
gradient, theta = sobel_filters(img_smoothed)  # 2. Compute magnitude and orientation of gradient
suppression_img = comput_Non_maximum_suppression(gradient, theta)  # 3. Compute Non-maximum suppression:
threshold_img, low_threshold, high_threshold = threshold(suppression_img)   # 4.1 Define two thresholds: low and high
final_img = myCannyEdgeDetector(threshold_img, Low_Threshold=low_threshold, High_Threshold=high_threshold)
# print(type(final_img), final_img)
print("--------------------------------------------\n")
print(low_threshold, high_threshold)
t1 = "Original color Image"
t2 = "edge output of myCannyEdgeDetector()"
plot_show_two_pic(inputImg_0, final_img, title1=t1, title2=t2)


##################################################################
# (iii) Compute the peak signal to noise ratio
print("--------------------------------------------\n")
print("(iii) Compute the peak signal to noise ratio")
# Compute the peak signal to noise ratio (PSNR) for an image.
skimage_psnr = skimage.measure.compare_psnr(feature_canny_1, final_img)
my_psnr = skimage.measure.compare_psnr(final_img, feature_canny_1)
str1 = "PSNR: skimages canny edge detector/my canny edge detector"
str2 = "PSNR: my canny edge detector/skimage  canny edge detector"
print(str1, str(skimage_psnr), "\n", str2, str(my_psnr))