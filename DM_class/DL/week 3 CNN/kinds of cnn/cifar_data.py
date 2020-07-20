from keras import datasets, models, layers
import numpy as np
import matplotlib.pyplot as plt


class Cifar10:
    def __init__(self,start, end):
        self.start = start
        self.end = end

    def get_cifar_data(self):
        (X_train_full, y_train_full), (X_test, y_test) = datasets.cifar10.load_data()
        X_train_full, X_test = X_train_full / 255.0, X_test / 255.0
        print(X_train_full.shape, y_train_full.shape)
        print(X_test.shape, y_test.shape)

        # shuffle the trianing dataset
        idxs = np.arange(X_train_full.shape[0])
        np.random.shuffle(idxs)
        X_train_full = X_train_full[idxs]
        y_train_full = y_train_full[idxs]
        X_train, y_train = X_train_full[0:self.start], y_train_full[0:self.start]
        X_valid, y_valid = X_train_full[self.start:self.end], y_train_full[self.start:self.end]
        print(X_train.shape, X_valid.shape)
        ##############################################################################
        # for test
        # X_train, y_train = X_train_full[0:100], y_train_full[0:100]
        # X_valid, y_valid = X_train_full[100:200], y_train_full[100:200]
        # X_test, y_test  = X_test[:100], y_test[:100]
        ##############################################################################
        return X_train, y_train, X_valid, y_valid, X_test, y_test

    def plot_images(self, X_train, y_train):
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
            plt.xlabel(class_names[y_train[i][0]])
        plt.show()




# c = Cifar10(50,100)
# X_train, y_train, X_valid, y_valid, X_test, y_test = c.get_cifar_data()
# c.plot_images(X_train, y_train)