# adopted from Skimage documentation
# adopted from Skimage documentation

from skimage import data  # To input standard images
from skimage.transform import rotate
from matplotlib import pyplot as plt  # To plot images
from skimage.color import rgb2gray  # To convery RGB image to grayscale
import numpy as np
from skimage.feature import local_binary_pattern  # Library for local_binary_pattern
from skimage.color import label2rgb

def part_1():
# settings for LBP
    radius = 3
    n_points = 8 * radius
    METHOD = 'uniform'  # method: {‘default’, ‘ror’, ‘uniform’, ‘var’}

    inputImage1 = data.astronaut()
    inputImage1 = rgb2gray(inputImage1)
    #Lbp computation

    inputImage2 = data.chelsea()
    inputImage2 = rgb2gray(inputImage2)
    #LBP computation
    lbp1 = local_binary_pattern(inputImage1, n_points, radius, method='uniform')
    lbp2 = local_binary_pattern(inputImage2, n_points, radius, method='var')
    #plotting
    fig, axarr = plt.subplots(2,2,figsize=(15,8))
    axarr[0,0].imshow(inputImage1,cmap='gray')
    axarr[0,0].set_title('Input image 1')
    axarr[0,1].imshow(inputImage2,cmap='gray')
    axarr[0,1].set_title('Input image 2')
    axarr[1,0].imshow(lbp1,cmap='gray')
    axarr[1,0].set_title('LBP for image 1')
    axarr[1,1].imshow(lbp2,cmap='gray')
    axarr[1,1].set_title('LBP for image 2')
    plt.show()


##################################################################################
def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    print(n_bins)
    return ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')


def overlay_labels(image, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)


def highlight_bars(bars, indexes):
    for i in indexes:
        bars[i].set_facecolor('r')


def part_2():
    radius = 3
    n_points = 8 * radius
    METHOD = 'uniform'
    image = data.brick()  # image
    # LBP Computation
    grayimage = rgb2gray(image)
    lbp = local_binary_pattern(grayimage, n_points, radius, method='uniform')
    # plot histograms of LBP of textures
    fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
    plt.gray()

    titles = ('edge', 'flat', 'corner')
    w = width = radius - 1
    edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
    flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))

    i_14 = n_points // 4  # 1/4th of the histogram
    i_34 = 3 * (n_points // 4)  # 3/4th of the histogram
    corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
                     list(range(i_34 - w, i_34 + w + 1)))

    label_sets = (edge_labels, flat_labels, corner_labels)

    for ax, labels in zip(ax_img, label_sets):
        ax.imshow(overlay_labels(image, lbp, labels))

    for ax, labels, name in zip(ax_hist, label_sets, titles):
        counts, _, bars = hist(ax, lbp)
        highlight_bars(bars, labels)
        ax.set_ylim(top=np.max(counts[:-1]))
        ax.set_xlim(right=n_points + 2)
        ax.set_title(name)

    ax_hist[0].set_ylabel('Percentage')
    for ax in ax_img:
        ax.axis('off')

    plt.show()

#####################################################################
def plt_images(a, b, c, d=None):
    fig, axarr = plt.subplots(2, 2, figsize=(10, 8))
    axarr[0, 0].imshow(a, cmap='gray')
    axarr[0, 0].set_title("brick")
    axarr[0, 1].imshow(b, cmap='gray')
    axarr[0, 1].set_title("grass")
    axarr[1, 0].imshow(c, cmap='gray')
    axarr[1, 0].set_title("gravel")
    plt.show()


def kullback_leibler_divergence(hist1,
                                hist2):  # Check how different two probability distributions are.. 0 means they are same
    hist1 = hist1.ravel()
    hist2 = hist2.ravel()
    return np.sum(np.where(hist1 != 0, hist1 * np.log(hist1 / hist2), 0))


def histogram_intersection_distance(hist1, hist2, widthHist=18):  # Compute Histogram Intersection based distance
    hist1 = hist1.ravel()
    hist2 = hist2.ravel()
    sumVal = 0
    for i in range(widthHist):
        sumVal += min(hist1[i], hist2[i])
    return sumVal


def hist(ax, lbp):  #
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')


def match(refs, img, distanceMetric):  # Match two histograms

    threshold_KL = 10.0  # choose a high number
    threshold_HistIntersection = 0.0

    lbp = local_binary_pattern(img, n_points, radius, METHOD)  # compute LBP map
    n_bins = int(lbp.max() + 1)

    hist, _ = np.histogram(lbp, density=True, bins=n_bins,
                           range=(0, n_bins))  # compute histogram from the LBP image map

    for name, ref in refs.items():
        ref_hist, _ = np.histogram(ref, density=True, bins=n_bins,
                                   range=(0, n_bins))
        if distanceMetric == 'histogram_intersection_distance':

            score = globals()[distanceMetric](hist, ref_hist,
                                              n_bins)  # use global to call function from global symbol table
            print(name,": Distance Metric: ", distanceMetric, " Score: ", score)

            if score > threshold_HistIntersection:  # comparison if we have found the best match
                threshold_HistIntersection = score
                labels = name

        if distanceMetric == 'kullback_leibler_divergence':
            score = globals()[distanceMetric](hist, ref_hist)
            print(name," :Distance Metric: ", distanceMetric, " Score: ", score)
            if score < threshold_KL:  # comparison if we have found the best match
                threshold_KL = score
                labels = name

    if distanceMetric == 'histogram_intersection_distance':
        return labels
    else:
        return labels


# define LBP, # settings for LBP
radius = 8
n_points = 8 * radius
METHOD = 'uniform' # discuss the different methods
#input images from data
brick = data.brick()
grass = data.grass()
gravel = data.gravel()
plt_images(brick, grass, gravel)
refs = {
    'brick': local_binary_pattern(brick, n_points, radius, METHOD),
    'grass': local_binary_pattern(grass, n_points, radius, METHOD),
    'gravel': local_binary_pattern(gravel, n_points, radius, METHOD)
}
fig, axarr = plt.subplots(2,2,figsize=(10,8))
axarr[0,0].imshow(refs["brick"],cmap='gray')
axarr[0,0].set_title("brick")
axarr[0,1].imshow(refs["grass"],cmap='gray')
axarr[0,1].set_title("grass")
axarr[1,0].imshow(refs["gravel"],cmap='gray')
axarr[1,0].set_title("gravel")
plt.show()
# classify rotated textures

print('Rotated images matched against references using LBP:')

print('original: brick, match result: ',
      match(refs, brick,'kullback_leibler_divergence'))
print('original: brick, match result: ',
      match(refs, brick,'histogram_intersection_distance'))
# plot histograms of LBP of textures

print('original: brick, rotated: 30deg, match result: ',
      match(refs, rotate(brick, angle=30, resize=False),'kullback_leibler_divergence')) #call the match() to compare the histograms of two images
print('original: brick, rotated: 70deg, match result: ',
      match(refs, rotate(brick, angle=70, resize=False),'kullback_leibler_divergence'))
print('original: grass, rotated: 145deg, match result: ',
      match(refs, rotate(grass, angle=145, resize=False),'kullback_leibler_divergence'))

# plot histograms of LBP of textures
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3,
                                                       figsize=(9, 6))
plt.gray()

ax1.imshow(brick)
ax1.axis('off')
hist(ax4, refs['brick'])
ax4.set_ylabel('Percentage')

ax2.imshow(grass)
ax2.axis('off')
hist(ax5, refs['grass'])
ax5.set_xlabel('Uniform LBP values')

ax3.imshow(gravel)
ax3.axis('off')
hist(ax6, refs['gravel'])

plt.show()