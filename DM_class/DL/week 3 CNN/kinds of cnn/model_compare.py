import tensorflow as tf
from resnet import ResNet
from tensorflow import keras
from keras import datasets, models, layers
import os, time
from cifar_data import Cifar10
from keras import backend as K
K.clear_session()  # Some memory clean-up

# os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0'  # use GPU

##########################################################################
# build 3 kind of cnn
def my_cnn():
    my_model = models.Sequential()
    my_model.add(layers.Conv2D(32, (9,9), padding="same",activation="relu", input_shape=(32,32,3)))
    my_model.add(layers.BatchNormalization(momentum=0.9))
    my_model.add(layers.Dropout(rate=0.25))

    my_model.add(layers.Conv2D(64, (9,9), padding="same",activation="relu"))
    my_model.add(layers.BatchNormalization(momentum=0.9))
    my_model.add(layers.MaxPool2D(pool_size=(2,2), padding="same"))
    my_model.add(layers.Dropout(rate=0.25))

    my_model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    my_model.add(layers.BatchNormalization(momentum=0.9))
    my_model.add(layers.MaxPool2D(pool_size=(2, 2), padding="same"))
    my_model.add((layers.Dropout(rate=0.25)))

    my_model.add(layers.Flatten())
    my_model.add(layers.Dense(512, activation="relu"))
    my_model.add(layers.BatchNormalization(momentum=0.9))
    my_model.add((layers.Dropout(rate=0.25)))
    my_model.add(layers.Dense(10, activation="softmax"))
    my_model.summary()
    return my_model

def lenet_5():
    leNet_5 = models.Sequential()
    leNet_5.add(layers.Conv2D(28, (5,5), padding="same", activation="tanh", input_shape=(32,32,3)))
    leNet_5.add(layers.AveragePooling2D(pool_size=(2,2), padding="same"))
    leNet_5.add(layers.Conv2D(10, (5,5), padding="same", activation="tanh"))
    leNet_5.add(layers.AveragePooling2D(pool_size=(2,2), padding="same"))
    leNet_5.add(layers.Conv2D(1, (5,5), padding="same", activation="tanh"))
    leNet_5.add(layers.Flatten())
    leNet_5.add(layers.Dense(84, activation="tanh"))
    leNet_5.add(layers.Dense(10, activation="softmax"))
    leNet_5.summary()
    return leNet_5

def alexnet():
    alexNet = models.Sequential()
    # alexNet.add(layers.Conv2D(55, (11,11), padding="same", activation="relu", input_shape=(224,224,3)))
    alexNet.add(layers.Conv2D(55, (11, 11), padding="same", activation="relu", input_shape=(32, 32, 3)))
    alexNet.add(layers.MaxPool2D(pool_size=(3,3), padding="valid"))
    alexNet.add(layers.Conv2D(27, (5,5), padding="same", activation="relu"))
    alexNet.add(layers.MaxPool2D(pool_size=(3,3), padding="valid"))
    alexNet.add(layers.Conv2D(13, (3,3), padding="same", activation="relu"))
    alexNet.add(layers.Conv2D(13, (3,3), padding="same", activation="relu"))
    alexNet.add(layers.Conv2D(13, (3,3), padding="same", activation="relu"))
    alexNet.add(layers.Flatten())
    alexNet.add(layers.Dense(4096, activation="relu"))
    alexNet.add(layers.Dense(4096, activation="relu"))
    #alexNet.add(layers.Dense(1000, activation="softmax"))
    alexNet.add(layers.Dense(10, activation="softmax"))
    alexNet.summary()
    return alexNet


def resnet():
    resNet = ResNet()
    resNet.build()
    resNet.summary()
    return resNet

if __name__ == '__main__':
    #my_cnn()
    #lenet_5()
    #alexnet()
    resnet()