import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import datasets
from keras import backend as K
from keras.backend import  manual_variable_initialization(True)
K.clear_session()  # Some memory clean-up


def small_data_try(x0, y0, x1, y1):
    #  can ignore, just small process
    batch = np.array([x0, x1], dtype=np.float32)
    print(batch.shape)

    model = keras.models.load_model('MyCNN.h5')
    # opt = keras.optimizers.SGD(lr=0.001, decay=0.01 / 40, momentum=0.9, nesterov=True)
    # model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_valid, y_valid))

    results = model.predict(batch, verbose=1)
    print(results.shape, "\n", y0[0], y1)
    print(np.argmax(results[0]), np.argmax(results[1]))


def plot_images(x, y ,class_names):
    # plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x, cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[y[0]])
    plt.show()


def plot_images(X_train, y_train=None, prediction=None, flag=None):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(8, 8))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_train[i], cmap=plt.cm.binary)
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        if flag:
            n = np.argmax(prediction[i])
            plt.xlabel(class_names[n])
        else:
            plt.xlabel(class_names[y_train[i][0]])
    plt.show()

def model_prediction(model_name, test_size):
    (X_train_full, y_train_full), (X_test, y_test) = datasets.cifar10.load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # X_train_full, X_test = X_train_full / 255.0, X_test / 255.0
    # X_train, y_train = X_train_full[0:1000], y_train_full[0:1000]
    # X_valid, y_valid = X_train_full[1000:2000], y_train_full[1000:2000]
    # print(y_test[:10])
    idxs = np.arange(X_test.shape[0])
    np.random.shuffle(idxs)
    X_test, y_test = X_test[idxs], y_test[idxs]
    # print(y_test[:10])


    model = keras.models.load_model(model_name)
    model.summary()
    # r = model.predict(X_test[:test_size], batch_size=32, verbose=1)
    opt = keras.optimizers.SGD(lr=0.001, decay=0.01 / 40, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1, batch_size=64)
    count = 0
    wrong_names = []
    # for i in range(test_size):
    #     if np.argmax(r[i]) == y_test[i][0]:
    #         count += 1
    #     else:
    #         wrong_names.append(class_names[y_test[i][0]])
    # print(count, wrong_names)

    # plot_images(X_test[:test_size], y_train=y_test[:test_size])
    # plot_images(X_test[:test_size], prediction=r, flag=True)

    #return count/test_size
    return test_loss, test_acc


if __name__ == '__main__':

    model_percent = {}
    name = ["AlexNet.h5", "LeNet5.h5", "MiniVGG_CNN_model.h5", "MiniVGG_CNN_model_augu.h5", "MyCNN.h5"]

    for n in name:
        K.clear_session()  # Some memory clean-up
        test_loss, test_acc = model_prediction(n, 10000)
        model_percent[n] = (test_loss, test_acc)

    print(model_percent)


