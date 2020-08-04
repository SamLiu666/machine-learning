from skimage import io, feature
from skimage.color import rgb2gray
import numpy as np
import math
from collections import defaultdict
from canny import canny_edge_detector

file_path = r"images\task2.jpg"
image = io.imread(file_path)
print(image.shape)

# turn into gray image
image_rgb = rgb2gray(image)
output = np.empty_like(image_rgb)
print(output.shape)
row, col = output.shape

# Find circles
rmin = 18
rmax = 20
steps = 100
threshold = 0.4

points = []
for r in range(rmin, rmax + 1):
    for t in range(steps):
        points.append((r, int(r * math.cos(2 * math.pi * t / steps)), int(r * math.sin(2 * math.pi * t / steps))))

acc = defaultdict(int)
for x, y in canny_edge_detector(image_rgb):
    for r, dx, dy in points:
        a = x - dx
        b = y - dy
        acc[(a, b, r)] += 1

circles = []
for k, v in sorted(acc.items(), key=lambda i: -i[1]):
    x, y, r = k
    if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
        print(v / steps, x, y, r)
        circles.append((x, y, r))

# for x, y, r in circles:
    # draw_result.ellipse((x-r, y-r, x+r, y+r), outline=(255,0,0,0))