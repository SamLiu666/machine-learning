import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, models, layers, regularizers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
bn_momentum= 0.9
bn_eps= 2E-5
reg= 0.001
DefaultBatchNorm = partial(keras.layers.BatchNormalization, momentum= bn_momentum, epsilon= bn_eps)
DefaultConv2D = partial(keras.layers.Conv2D, kernel_regularizer= regularizers.l2(reg), use_bias= False, padding = 'same')


class ResNet:
    def __init__(self, num_classes=10, batch_size=32, num_epochs=20, optimizer='adam', learning_rate=0.001,
                 verbose=True, random_state=42):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optimizer = keras.optimizers.get(optimizer)
        self.optimizer.learning_rate = learning_rate
        self.random_state = random_state
        self.verbose = verbose
        keras.backend.clear_session()
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
        self.model = None

    @staticmethod
    def ResidualModule(inputs, K=64, strides=(1, 1), skip_connection=False):
        # bulid residualmodul
        main_layers = [DefaultBatchNorm(), layers.Activation(activation='relu'),
                       DefaultConv2D(filters=int(K / 4), kernel_size=1, strides=(1, 1)),
                       DefaultBatchNorm(), layers.Activation(activation='relu'),
                       DefaultConv2D(filters=int(K / 4), kernel_size=3, strides=strides),
                       DefaultBatchNorm(), layers.Activation(activation='relu'),
                       DefaultConv2D(filters=K, kernel_size=1, strides=(1, 1))]

        skip_layers = []
        if skip_connection:
            skip_layers = [DefaultBatchNorm(), layers.Activation(activation='relu'),
                           DefaultConv2D(filters=K, kernel_size=1, strides=strides)]

        h = inputs
        for layer in main_layers:
            h = layer(h)
        short_cut = inputs
        for layer in skip_layers:
            short_cut = layer(short_cut)

        return (h + short_cut)

    def build(self, blocks=[3, 4], filters=[16, 16, 16]):
        self.model = models.Model()
        inputs = layers.Input(shape=(32, 32, 3))
        h = inputs
        h = DefaultBatchNorm()(h)
        h = DefaultConv2D(filters=filters[0], kernel_size=3)(h)

        for i in range(len(blocks)):
            strides = (1, 1) if i == 0 else (
            2, 2)  # We downsample at the begining residual module of each block except the first block
            h = ResNet.ResidualModule(h, filters[i], strides,
                                      True)  # apply the skip connection on the first module of the block

            for j in range(1, blocks[i] - 1, 1):  # Add more blocks[i]-1 residual models
                h = ResNet.ResidualModule(h, filters[i], (1, 1), False)  # no skip connection for these residual module

        h = DefaultBatchNorm()(h)
        h = layers.Activation(activation='relu')(h)
        h = layers.AveragePooling2D(pool_size=(8, 8))(h)
        h = layers.Flatten()(h)
        h = layers.Dense(units=self.num_classes, activation="softmax")(h)
        self.model = models.Model(inputs=inputs, outputs=h, name="ResNet")  # We now have a ResNet model
        self.model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, batch_size=None, num_epochs=None, verbose=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        num_epochs = num_epochs if num_epochs is not None else self.num_epochs
        verbose = verbose if verbose is not None else self.num_epochs
        self.history = self.model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=batch_size,
                                      epochs=num_epochs, verbose=1 if verbose else 0)

    def evaluate(self, X_test, y_test):
        loss, acc = self.model.evaluate(X_test, y_test)
        return acc, loss

    def plot_progress(self):
        pd.DataFrame(self.history.history).plot(figsize=(8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, np.max(self.history.history['loss']))  # Set the vertical range to [0-max(train loss)]
        plt.show()

    def summary(self):
        print(self.model.summary())

    def save(self):
        self.model.save("ResNet.h5")