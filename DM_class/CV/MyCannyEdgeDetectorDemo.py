from skimage import data
from skimage import io
import numpy as np
from skimage.color import *
import matplotlib.pyplot as plt
from skimage import feature
import skimage
from skimage.feature import corner_harris, corner_subpix, corner_peaks # Library for Harris corner detection

def  myCannyEdgeDetector(image, Low_Threshold, High_Threshold):
    """:arg
    """
    width, height, depth = image.shape
    segmented_image = np.zeros([width,height]) # Create an empty array of same size as InputImage
    for iRow in range(width): # Loop corresponding to the rows
        for iCol in range(height): # Loop corresponding to the column
            # Check the pixel value if it lies in the skin color range
            if(image[iRow,iCol,0]>Low_Threshold and image[iRow,iCol,0]<High_Threshold):
                segmented_image[iRow,iCol] = 1
            else:
                segmented_image[iRow,iCol] = 0
    return segmented_image


def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / wid)

inputImg = io.imread('3.jpg')
width, height, depth = inputImg.shape
# inputImg = gaussian(width, height, depth,1) #blur filter - Play with the value of the filter

#  (ii) edge output of myCannyEdgeDetector()
segmented_image = myCannyEdgeDetector(inputImg*255, Low_Threshold=80, High_Threshold=236)

width, height = segmented_image.shape

# (i) output of skimage.feature.canny for the input image
print("--------------------------------------------\n")
print("(i) output of skimage.feature.canny for the input image")
edgeMap1 = feature.canny(segmented_image, sigma=1)
edgeMap2 = feature.canny(segmented_image, sigma=3) # different sigma value
fig, axes = plt.subplots(1, ncols=2, figsize=(8, 4))
axes[0].imshow(edgeMap1)
axes[0].set_title("skimage.feature.canny with sigma=1")
axes[1].imshow(edgeMap2)
axes[1].set_title("skimage.feature.canny with sigma=3")
plt.show()


# (ii)edge output of myCannyEdgeDetector();
print("--------------------------------------------\n")
print("(ii)edge output of myCannyEdgeDetector()")
# ax[0].imshow(inputImg)
plt.imshow(segmented_image,cmap='gray')
plt.title("(ii)edge output of myCannyEdgeDetector()")
plt.show()

# (iii) Compute the peak signal to noise ratio
print("--------------------------------------------\n")
print("(iii) Compute the peak signal to noise ratio")
corner_Peaks_0 = corner_peaks(inputImg, min_distance=5 , threshold_rel=0.09)
corner_Peaks_1 = corner_peaks(segmented_image, min_distance=5 , threshold_rel=0.09)
fig, axes = plt.subplots(1, ncols=2, figsize=(15, 10))
axes = axes.ravel()
axes[0].imshow(corner_Peaks_0)
axes[0].set_title("inputImg, min_distance=5 , threshold_rel=0.09")
axes[1].imshow(corner_Peaks_1)
axes[1].set_title("segmented_image, min_distance=5 , threshold_rel=0.09")
plt.show()


