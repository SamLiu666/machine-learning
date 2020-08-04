import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data, color, io
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte


# Load picture and detect edges
file_path = r"images\task2.jpg"
image = io.imread(file_path)
image_rgb = rgb2gray(image)
edges = canny(image_rgb, sigma=1, low_threshold=10, high_threshold=50)

# Detect two radii
hough_radii = np.arange(0, 50, 2)
hough_res = hough_circle(edges, hough_radii)

# Select the most prominent 3 circles
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=3)

# Draw them
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
image = color.gray2rgb(image)
for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_y, center_x, radius,
                                    shape=image.shape)
    image[circy, circx] = (220, 20, 20)

ax.imshow(image)
plt.axis('off')
plt.show()