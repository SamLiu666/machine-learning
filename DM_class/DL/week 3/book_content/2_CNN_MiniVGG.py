import csv
import tensorflow as tf
from tensorflow import keras
from keras import datasets, models, layers
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
import os, time
from resnet import ResNet
from keras.preprocessing.image import ImageDataGenerator
os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0'  # use GPU
K.clear_session()  # Some memory clean-up


def plot_images(X_train, y_train):
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


def plot_loss_accuracy(history):
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(history.history['accuracy'], label='train accuracy', color='green', marker="o")
    ax1.plot(history.history['val_accuracy'], label='valid accuracy', color='blue', marker = "v")
    ax2.plot(history.history['loss'], label = 'train loss', color='orange', marker="o")
    ax2.plot(history.history['val_loss'], label = 'valid loss', color='red', marker = "v")
    ax1.legend(loc=3)

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy', color='g')
    ax2.set_ylabel('Loss', color='b')
    ax2.legend(loc=4)
    plt.ylim([0.6, 2.5])
    plt.show()


def load_cifar_data():
    (X_train_full, y_train_full), (X_test, y_test) = datasets.cifar10.load_data()
    X_train_full, X_test = X_train_full / 255.0, X_test / 255.0
    print(X_train_full.shape, y_train_full.shape)
    print(X_test.shape, y_test.shape)

    # shuffle the trianing dataset
    idxs = np.arange(X_train_full.shape[0])
    np.random.shuffle(idxs)
    X_train_full = X_train_full[idxs]
    y_train_full = y_train_full[idxs]
    X_train, y_train = X_train_full[0:20000], y_train_full[0:20000]
    X_valid, y_valid = X_train_full[20000:40000], y_train_full[20000:40000]

    ##############################################################################
    # for test
    # X_train, y_train = X_train_full[0:100], y_train_full[0:100]
    # X_valid, y_valid = X_train_full[100:200], y_train_full[100:200]
    # X_test, y_test  = X_test[:100], y_test[:100]
    ##############################################################################
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def build_model():
    vgg_model = models.Sequential()
    vgg_model.add(layers.Conv2D(32, (3,3), padding="same",activation="relu", input_shape=(32,32,3)))
    vgg_model.add(layers.BatchNormalization(momentum=0.9))

    vgg_model.add(layers.Conv2D(32, (3,3), padding="same",activation="relu"))
    vgg_model.add(layers.BatchNormalization(momentum=0.9))

    vgg_model.add(layers.MaxPool2D(pool_size=(2,2), padding="same"))
    vgg_model.add((layers.Dropout(rate=0.25)))

    vgg_model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    vgg_model.add(layers.BatchNormalization(momentum=0.9))

    vgg_model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    vgg_model.add(layers.BatchNormalization(momentum=0.9))

    vgg_model.add(layers.MaxPool2D(pool_size=(2, 2), padding="same"))
    vgg_model.add((layers.Dropout(rate=0.25)))

    vgg_model.add(layers.Flatten())
    vgg_model.add(layers.Dense(512, activation="relu"))
    vgg_model.add(layers.BatchNormalization(momentum=0.9))
    vgg_model.add((layers.Dropout(rate=0.25)))

    vgg_model.add(layers.Dense(10, activation="softmax"))
    # vgg_model.summary()
    return vgg_model


def trian(vgg_model, X_train, y_train, X_valid, y_valid, X_test, y_test, aumentation_flag=None):
    # batch_size = 32 (default)
    opt = keras.optimizers.SGD(lr=0.001, decay=0.01 / 40, momentum=0.9, nesterov=True)
    vgg_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    if aumentation_flag:
        # II. MiniVGG with data augmentation
        aug = ImageDataGenerator(rotation_range=5, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1,
                                 zoom_range=0.1, horizontal_flip=True, fill_mode="nearest")
        history = vgg_model.fit_generator(aug.flow(x=X_train, y=y_train, batch_size=32), epochs=20,
                                          validation_data=(X_valid, y_valid))
    else:
        history = vgg_model.fit(x=X_train, y=y_train, batch_size=32, epochs=20, validation_data=(X_valid, y_valid))

    test_loss, test_acc = vgg_model.evaluate(X_test, y_test, verbose=1, batch_size=64)
    print("Test acc is {}".format(test_acc))
    return history, test_loss, test_acc


if __name__ == '__main__':
    cost_time, acc, loss = {}, {}, {}  # save training results
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_cifar_data()
    ##########################################################################
    print("------------------------------------------------------------\n")
    print("trian model")
    vgg_model = build_model()
    #vgg_model.summary()
    plot_images(X_train, y_train)

    start = time.time()
    history, test_loss, test_acc = trian(vgg_model, X_train, y_train, X_valid, y_valid, X_test, y_test)
    end = time.time()
    # print("total time: ", end - start)
    cost_time["MiniVGG"]= end - start
    loss["MiniVGG"] = test_loss
    acc["MiniVGG"] = test_acc
    plot_loss_accuracy(history)

    # save and reload model
    vgg_model.save("MiniVGG_CNN_model.h5")
    new_model = keras.models.load_model('MiniVGG_CNN_model.h5')
    new_model.summary()

    #######################################################################################
    # data augumentation train
    # K.clear_session()  # Some memory clean-up
    print("------------------------------------------------------------\n")
    print("trian model")
    start = time.time()
    history_augu, test_loss_augu, test_acc_augu = trian(vgg_model, X_train, y_train, X_valid, y_valid, X_test, y_test, aumentation_flag=True)
    end = time.time()
    # print("total time: ", end - start)
    cost_time["MiniVGG_data_aug"]= end - start
    loss["MiniVGG_data_aug"] = test_loss_augu
    acc["MiniVGG_data_aug"] = test_acc_augu

    # save,reload, and show model
    plot_loss_accuracy(history_augu)
    vgg_model.save("MiniVGG_CNN_model_augu.h5")
    print("Data augmentaion differnece: ", test_loss_augu-test_loss)

    #######################################################################################
    #  resnet
    res_net = ResNet()
    res_net.build()
    res_net.summary()
    start = time.time()
    res_net.fit(X_train, y_train, X_valid, y_valid, batch_size=32, num_epochs=20)
    test_acc_resnet, test_loss_resnet = res_net.evaluate(X_test, y_test)
    end = time.time()
    #print("total time: ", end - start)
    cost_time["ResNet"]= end - start
    loss["ResNet"] = test_loss_resnet
    acc["ResNet"] = test_acc_resnet
    # res.save

    # new_model_1 = keras.models.load_model("ResNet.h5")  # json error
    # new_model_1.summary()
    res_net.evaluate(X_test, y_test)
    res_net.plot_progress()
    ########################################################################################
    print("------------------results\n")
    print("cost_time:  ", cost_time)
    print("loss:  ", loss)
    print("accuracy:  ", acc)
    # res_net.save_model()

    cost_time["record"]= "time"
    loss["record"] = "loss"
    acc["record"] = "accuracy"

    # write to txt
    data = [cost_time, loss, acc]
    with open('model.txt', 'a', newline='', encoding='utf-8') as f:
        for d in data:
            for i, j in d.items():
                f.writelines((i + "   " + str(j) + "    "))
            f.write("\n")
        f.close()
    print("\n all done")