import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.image as mpimg

model = keras.models.load_model("image_classify_modify.h5")
model.summary()

data = mpimg.imread("1.jpg")
print(data.shape)