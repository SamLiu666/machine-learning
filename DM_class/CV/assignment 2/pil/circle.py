from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin
from canny import canny_edge_detector
from MyCannyEdgeDetectorDemo import myCannyEdgeDetector,gaussian,convolve,sobel_filters,comput_Non_maximum_suppression
from collections import defaultdict
import matplotlib.pyplot as plt
from skimage import color, io,feature
import numpy as np


input_image = io.imread("input.jpg")   # Load image
input_image_gray = color.rgb2gray(input_image)   # rgb2gray
input_image_canny = feature.canny(input_image_gray, sigma=1)   # edge detector, also can use the assignment1
print(input_image_canny.shape)
output = denoise_bilateral(input_image_canny, sigma_color=0.01, sigma_spatial=5, multichannel=True)

plt.imshow(input_image_canny)
plt.title("canny")
plt.show()

#output_image = np.zeros_like(input_image)
output_image = np.zeros_like(input_image_gray)
print(output_image.shape)

# define circles
rmin = 18
rmax = 20
# rmax = 100
steps = 100
threshold = 0.4

points = []
for r in range(rmin, rmax + 1):
    for t in range(steps):
        points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))

acc = defaultdict(int)
for x, y in canny_edge_detector(input_image):
# for x, y in final_img:
    for r, dx, dy in points:
        a = x - dx
        b = y - dy
        #acc[(a, b, r)] += 1
        acc[(a, b, r)] += 1

circles = []
count = 0
for k, v in sorted(acc.items(), key=lambda i: -i[1]):
    x, y, r = k
    if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
        print(v / steps, x, y, r)
        circles.append((x, y, r))
        count += 1
for x, y, r in circles:
    draw_result.ellipse((x-r, y-r, x+r, y+r), outline=(255,0,0,0))

print("How many circles: ",count)
# Save output image
plt.imshow(output_image)
plt.show()
output_image.save("result.png")