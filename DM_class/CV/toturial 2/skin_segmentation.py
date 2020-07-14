from skimage import data
from skimage import io
import numpy as np
from skimage.color import *
import matplotlib.pyplot as plt

def Segmented_image (inputImage):
    width, height, depth = inputImage.shape
    segmented_image = np.zeros([width,height]) # Create an empty array of same size as InputImage
    for iRow in range(width): # Loop corresponding to the rows
        for iCol in range(height): # Loop corresponding to the column
            # Check the pixel value if it lies in the skin color range
            if(inputImage[iRow,iCol,0]>80 and inputImage[iRow,iCol,0]<236): # Across red channel
                if(inputImage[iRow,iCol,1]>47 and inputImage[iRow,iCol,1]<188): # Across green channel
                    if(inputImage[iRow,iCol,2]>42 and inputImage[iRow,iCol,2]<180):# Across blue channel
                            segmented_image[iRow,iCol] = 1
            else:
                segmented_image[iRow,iCol] = 0
    return segmented_image


def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / wid)

inputImg = io.imread('3.jpg')
# width, height, depth = input_Image.shape
# inputImg = gaussian(width, height, depth,1) #blur filter - Play with the value of the filter
segmented_image = Segmented_image(inputImg*255)
fig, axes = plt.subplots(1, ncols=2, figsize=(8, 4))
ax = axes.ravel()
ax[0].imshow(inputImg)
ax[1].imshow(segmented_image,cmap='gray')
plt.show()