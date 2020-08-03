from skimage import io
from skimage.color import rgb2gray
import numpy as np
from math import sqrt

file_path = r"images\task2.jpg"
image = io.imread(file_path)
print(image.shape)
image_rgb = rgb2gray(image)

output = np.empty_like(image)
print(output.shape)