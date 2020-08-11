# from impy import imarray
import numpy as np
import matplotlib.pyplot as plt
# from scipy.ndimage import imread
from skimage import io, color, feature
from MyCannyEdgeDetectorDemo import *
from scipy.ndimage.filters import convolve
import time
from skimage.draw import circle_perimeter
from skimage import draw


def my_canny_edge(img_gray):
    # my_canny_edge detector from assignment1
    img_gaussian = gaussian(k=5, sigma=1)  # 1. Gaussian  with kenel=5x5
    img_smoothed = convolve(img_gray, img_gaussian)  # Gaussian smooth
    gradient, theta = sobel_filters(img_smoothed)  # 2. Compute magnitude and orientation of gradient
    suppression_img = comput_Non_maximum_suppression(gradient, theta)  # 3. Compute Non-maximum suppression:
    threshold_img, low_threshold, high_threshold = threshold(suppression_img)  # 4.1 Define two thresholds: low and high
    final_img = myCannyEdgeDetector(threshold_img, Low_Threshold=low_threshold, High_Threshold=high_threshold)
    return final_img


def draw_circle(image, circles, name, path):
    # draw the circle on the original image
    print("draw image")

    for i in range(len(circles)):
        x,y,radius = circles[i]
        rr, cc = draw.circle_perimeter(x,y,radius)  # draw circle
        draw.set_color(image, [rr, cc], [255,255,0])  # yellow
    plt.title(name)
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(path)
    plt.show()
    print("Image saved")


def HoughCircles(input, circles, radius):
    # image shape
    rows = input.shape[0]
    cols = input.shape[1]

    # initialize and save the angles for computation
    sinang = dict()
    cosang = dict()
    for angle in range(0, 360):
        sinang[angle] = np.sin(angle * np.pi / 180)
        cosang[angle] = np.cos(angle * np.pi / 180)

    for r in radius:
        print("###################### radius: ", r)  # show processing
        # Initialize zeroes 2D array to save pixel
        acc_cells = np.full((rows, cols), fill_value=0, dtype=np.uint64)

        # loop the input
        for x in range(rows):
            #print("########## processing rows: ", x)  # how processing
            for y in range(cols):
                if input[x][y] == 255:  # it is the edge
                    for angle in range(0, 360):
                        # do the Using the formula
                        a = int(x - round(r * cosang[angle]))
                        b = int(y - round(r * sinang[angle]))
                        if a >= 0 and a < rows and b >= 0 and b < cols:
                            acc_cells[a][b] += 1

        print('## do radius: ', r)
        acc_cell_max = np.amax(acc_cells)
        #print('## max acc value: ', acc_cell_max)
        if (acc_cell_max > 150):
            #print("Detecting the circles for radius: ", r)
            # Initial threshold
            acc_cells[acc_cells < 150] = 0
            # find the circles based on the radius
            for i in range(rows):
                for j in range(cols):
                    if (i > 0 and j > 0 and i < rows - 1 and j < cols - 1 and acc_cells[i][j] >= 150):
                        avg_sum = np.float32((acc_cells[i][j] + acc_cells[i - 1][j] + acc_cells[i + 1][j] +
                                              acc_cells[i][j - 1] + acc_cells[i][j + 1] + acc_cells[i - 1][j - 1] +
                                              acc_cells[i - 1][j + 1] + acc_cells[i + 1][j - 1] + acc_cells[i + 1][
                                                  j + 1]) / 9)
                        #print("Intermediate avg_sum: ", avg_sum)
                        if (avg_sum >= 33):
                            #print("For radius: ", r, "average: ", avg_sum, "\n")
                            print("x,y,radius: ",i,j,r)
                            circles.append((i, j, r))  # save the
                            acc_cells[i:i + 5, j:j + 7] = 0
    return circles

def CountShapes_ID(image):

    print("############################################################")
    print("1. read original image and turn it into gray ")
    img_gray = color.rgb2gray(image)
    print(image.shape, img_gray.shape)

    print("############################################################")
    print("2. use my canny edge to get the edge image ")
    # img_edge = feature.canny(img_gray, sigma=1.5)   # canny from system
    img_edge = my_canny_edge(img_gray)  # my canny edge detector from assignment 1
    # plt.imshow(img_edge)
    # plt.show()
    print("canny edge image: ",img_edge,img_edge.shape)

    print("############################################################")
    print("3. do the hough transfrom ")
    circles_temp = []  # save the (x,y,radius)
    # initializethe different radius based on the input
    #  use the common seting 10,70; not the whole row for convience test
    # radius_range = [i for i in range(10, 15)]
    radius_range = [i for i in range(15, 76, 20)]
    Circles = HoughCircles(img_edge, circles_temp, radius=radius_range)  # count circles return [x,y,radius]

    # save path and isplay the output image
    name = "Hough Transform"
    path = r"images\output\output_task2.jpg"
    draw_circle(image, Circles, name, path)


if __name__ == '__main__':
    start = time.time()  # for loop time record

    file_path = r"images\task2.jpg"
    image = io.imread(file_path)  # read the input image
    CountShapes_ID(image)  # count circles

    end = time.time()
    print("cost time: ", end - start)