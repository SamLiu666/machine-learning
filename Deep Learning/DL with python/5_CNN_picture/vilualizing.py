import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0'
# 1 activation visualization
path = r"D:\machine learning\Deep Learning\DL with python\5_CNN_picture\cats_and_dogs_small_2.h5"
model = load_model(path)
model.summary()

img_path = r"E:\chrome download\paper\corpus\train\train\cats_and_dogs_small\test\cats\cat.1700.jpg"
img = image.load_img(img_path, target_size=(150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

# Its shape is (1, 150, 150, 3)
print(img_tensor.shape)
plt.show(img_tensor[0])
plt.show()

from keras import models

# Extracts the outputs of the top 8 layers:
layer_outputs = [layer.output for layer in model.layers[:8]]
# Creates a model that will return these outputs, given the model input:
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
print(first_layer_activation.shape)
import matplotlib.pyplot as plt

plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
plt.show()