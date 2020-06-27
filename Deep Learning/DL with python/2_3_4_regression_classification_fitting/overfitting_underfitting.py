import keras
from keras import backend as K
from keras import models
from keras import layers
from keras import regularizers

# Some memory clean-up
K.clear_session()

from keras.datasets import imdb
import numpy as np

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results


# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)
# Our vectorized labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

def build_models(flag=True):
    # flag 为真，大网络
    if flag:
        original_model = models.Sequential()
        original_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                                        activation='relu', input_shape=(10000,)))
        original_model.add(layers.Dropout(0.5))
        original_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                                        activation='relu'))
        original_model.add(layers.Dropout(0.5))
        original_model.add(layers.Dense(1, activation='sigmoid'))

        original_model.compile(optimizer='rmsprop',
                               loss='binary_crossentropy',
                               metrics=['acc'])
        original_hist = original_model.fit(x_train, y_train,
                                           epochs=20,
                                           batch_size=512,
                                           validation_data=(x_test, y_test))
        print("big net work")
        return original_hist
    else:
        smaller_model = models.Sequential()
        smaller_model.add(layers.Dense(4, kernel_regularizer=regularizers.l2(0.001),
                                       activation='relu', input_shape=(10000,)))
        smaller_model.add(layers.Dense(4, kernel_regularizer=regularizers.l2(0.001),
                                       activation='relu'))
        smaller_model.add(layers.Dense(1, activation='sigmoid'))

        smaller_model.compile(optimizer='rmsprop',
                              loss='binary_crossentropy',
                              metrics=['acc'])

        smaller_model_hist = smaller_model.fit(x_train, y_train,
                                               epochs=20,
                                               batch_size=512,
                                               validation_data=(x_test, y_test))
        print("small net work")
        return smaller_model_hist


epochs = range(1, 21)
original_hist  = build_models(flag=True)
smaller_model_hist = build_models(flag=False)

history_dict = original_hist.history
print(history_dict.keys())

epochs = range(1, 21)
original_val_loss = original_hist.history['val_loss']
smaller_model_val_loss = smaller_model_hist.history['val_loss']

import matplotlib.pyplot as plt

# b+ is for "blue cross"
plt.plot(epochs, original_val_loss, 'b+', label='Original model')
# "bo" is for "blue dot"
plt.plot(epochs, smaller_model_val_loss, 'bo', label='Smaller model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()
