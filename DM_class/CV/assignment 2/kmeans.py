import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()


# the euclidean distance as measurement.
"""First we initialize k points, called means, randomly.
We categorize each item to its closest mean and we update the mean’s coordinates,
which are the averages of the items categorized in that mean so far.
We repeat the process for a given number of iterations and at the end, we have our clusters."""



def plot_points(points):
    plt.scatter(points[:, 0], points[:, 1])
    ax = plt.gca()
    ax.add_artist(plt.Circle(np.array([1, 0]), 0.75/2, fill=False, lw=3))
    ax.add_artist(plt.Circle(np.array([-0.5, 0.5]), 0.25/2, fill=False, lw=3))
    ax.add_artist(plt.Circle(np.array([-0.5, -0.5]), 0.5/2, fill=False, lw=3))
    plt.show()


def tag_points(points):
    plt.scatter(points[:, 0], points[:, 1])
    centroids = initialize_centroids(points, 3)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100)
    plt.show()


def initialize_centroids(points, k):
    # 1. create k central points randomly
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]

def closest_centroid(points, centroids):
    # 2. find the closest centroid for each point, use numpy broadcast
    distance = np.sqrt((points - centroids[:, np.newaxis])**2).sum(axis=2)
    print(distance.shape)
    return np.argmax(distance, axis=0)

def move_centroids(points, closest, centroids):

    return np.array([points[closest == k].mean(axis=0) for k in range(centroids.shape[0])])

def kkmeans(points):
    centroids = initialize_centroids(points, 3)  # 3 points
    print(points.shape, centroids.shape, centroids)
    tag_points(points)

    closest = closest_centroid(points, centroids)
    print(closest)

    results = move_centroids(points, closest, centroids)
    print(results)

def kMeans(X, K, maxIters = 10, plot_progress = None):
    # kmeans for 2 dimension
    centroids = X[np.random.choice(np.arange(len(X)), K), :]
    for i in range(maxIters):
        # Cluster Assignment step
        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
        # Move centroids step
        centroids = [X[C == k].mean(axis = 0) for k in range(K)]
        if plot_progress != None:
            plot_progress(X, C, np.array(centroids))
    return np.array(centroids) , C


points = np.vstack(((np.random.randn(150, 2) * 0.75 + np.array([1, 0])),
                  (np.random.randn(50, 2) * 0.25 + np.array([-0.5, 0.5])),
                  (np.random.randn(50, 2) * 0.5 + np.array([-0.5, -0.5]))))
print(points.shape)
centroids, C = kMeans(points, 3, maxIters = 10, plot_progress = None)
print(centroids.shape, centroids, C.shape, C, sep="\n")