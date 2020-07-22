import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
sns.set_context("notebook")
sns.set_style("darkgrid")


def return_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection


mu_1 = -4 # mean of first distribution
mu_2 = 4  # mean of the second distribution
data_1 = np.random.normal(mu_1, 2.0, 1000)
data_2 = np.random.normal(mu_2, 2.0, 1000)
hist_1, _ = np.histogram(mu_1, bins=100, range=[-15,15])
hist_2, _ = np.histogram(mu_2, bins=100, range=[-15,15])
intersection = return_intersection(hist_1, hist_2)

plt.hist(intersection)
plt.show()

# %% Plot distributions on their own axis
sns.jointplot(x=hist_1, y=hist_2, kind="kde", space=0)
plt.show()