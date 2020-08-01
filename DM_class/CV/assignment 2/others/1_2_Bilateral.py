import numpy as np
import skimage, math
from skimage import io, img_as_float
import matplotlib.pyplot as plt
from skimage.restoration import denoise_bilateral

def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)


def gaussian(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))


def apply_bilateral_filter(source, filtered_image, x, y, diameter, sigma_i, sigma_s):
    hl = diameter/2
    i_filtered = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        while j < diameter:
            neighbour_x = x - (hl - i)
            neighbour_y = y - (hl - j)
            if neighbour_x >= len(source):
                neighbour_x -= len(source)
            if neighbour_y >= len(source[0]):
                neighbour_y -= len(source[0])
            gi = gaussian(source[neighbour_x][neighbour_y] - source[x][y], sigma_i)
            gs = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_s)
            w = gi * gs
            i_filtered += source[neighbour_x][neighbour_y] * w
            Wp += w
            j += 1
        i += 1
    i_filtered = i_filtered / Wp
    filtered_image[x][y] = int(round(i_filtered))


def bilateral_filter_own(source, filter_diameter, sigma_i, sigma_s):
    filtered_image = np.zeros(source.shape)
    i = 0
    while i < len(source):
        j = 0
        while j < len(source[0]):
            apply_bilateral_filter(source, filtered_image, i, j, filter_diameter, sigma_i, sigma_s)
            j += 1
        i += 1
    return filtered_image


from skimage.restoration import denoise_bilateral as bilateralfilter


def bilateral(image, sigmaspatial, sigmarange):
    """
    :param image: np.array
    :param sigmaspatial: float || int
    :param sigmarange: float || int
    :return: np.array
    The 'image' 'np.array' must be given gray-scale. It is suggested that to use scikit-image.
    """

    return bilateralfilter(image, win_size=10, sigma_color=sigmarange,
                           sigma_spatial=sigmaspatial, bins=1000, multichannel=False)


def plot_show_two_pic(one, two, title1=None, title2=None):
    # plot two images for convience
    fig, axes = plt.subplots(1, ncols=2, figsize=(15, 8))

    axes[0].imshow(one, cmap='gray')
    axes[0].set_title(title1)
    plt.axis('off')

    axes[1].imshow(two, cmap='gray')
    axes[1].set_title(title2)
    plt.axis('off')
    plt.show()


image = r"images\task1.jpg"  # image path
src = io.imread(image)
# use float for standard deviation
src_image = img_as_float(io.imread(image))
output_1 = denoise_bilateral(src_image, sigma_color=0.025,
                             sigma_spatial=1.5, multichannel=True)
plot_show_two_pic(src, output_1)
output_1_2 = denoise_bilateral(output_1, sigma_color=0.05,
                             sigma_spatial=1.5, multichannel=True)
plot_show_two_pic(output_1, output_1_2)
output_1_2_2 = denoise_bilateral(output_1_2, sigma_color=0.01,
                             sigma_spatial=1.5, multichannel=True)
plot_show_two_pic(output_1_2, output_1_2_2)