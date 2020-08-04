from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin
from canny import canny_edge_detector
from MyCannyEdgeDetectorDemo import myCannyEdgeDetector,gaussian,convolve,sobel_filters,comput_Non_maximum_suppression
from collections import defaultdict
import matplotlib.pyplot as plt
from skimage import color, io,feature
import numpy as np


# Load image:
# input_image = Image.open("input.jpg")
# input_image_rgb = color.rgb2gray(input_image__1)  # rgb2gray
input_image = io.imread("input.jpg")
input_image_gray = color.rgb2gray(input_image)   # rgb2gray
input_image_canny = feature.canny(input_image_gray, sigma=1)
plt.imshow(input_image_canny)
plt.title("canny")
plt.show()

# Output image:
# output_image = Image.new("RGB", input_image.size)
# output_image.paste(input_image)
# draw_result = ImageDraw.Draw(output_image)

output_image = np.zeros_like(input_image)
print(output_image.shape)
# Find circles
rmin = 18
# rmax = 20
rmax = 100
steps = 100
threshold = 0.4

points = []
for r in range(rmin, rmax + 1):
    for t in range(steps):
        points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))

# my canny edge detector from assignment 1
# img_gaussian = gaussian(k=5, sigma=1)  # 1. Gaussian  with kenel=5x5
# img_smoothed = convolve(input_image_1, img_gaussian)  # Gaussian smooth
# gradient, theta = sobel_filters(img_smoothed)  # 2. Compute magnitude and orientation of gradient
# suppression_img = comput_Non_maximum_suppression(gradient,theta)  # 3. Compute Non-maximum suppression:
# threshold_img, low_threshold, high_threshold = threshold(suppression_img)  # 4.1 Define two thresholds: low and high
# final_img = myCannyEdgeDetector(threshold_img, Low_Threshold=low_threshold, High_Threshold=high_threshold)


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