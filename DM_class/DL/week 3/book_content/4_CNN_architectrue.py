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
    #leNet_5.summary()
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
    #alexNet.summary()
    return alexNet


def resnet():
    resNet = ResNet()
    resNet.build()
    #resNet.summary()
    return resNet


##########################################################################
# train
def load_data(start, end):
    # load cifar10 images dataset
    dataset = Cifar10(start, end)
    X_train, y_train, X_valid, y_valid, X_test, y_test = dataset.get_cifar_data()
    #dataset.plot_images(X_train, y_train)
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def get_model(all_models):
    # put all models in to a list for convience
    LeNet5 = lenet_5()
    all_models.append(LeNet5)

    AlexNet = alexnet()
    all_models.append(AlexNet)

    MyCNN = my_cnn()
    all_models.append(MyCNN)
    #AlexNet.summary()
    #print(all_models)
    # for model in all_models:
    #     model.summary()
    return all_models


def compile_mode(model, X_train, y_train, X_valid, y_valid, X_test, y_test):
    # trian the model
    opt = keras.optimizers.SGD(lr=0.001, decay=0.01 / 40, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_valid, y_valid))
    #model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_valid, y_valid))
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1, batch_size=64)
    print("Test acc is {}".format(test_acc))
    return history, test_loss, test_acc


def model_save(model, name):
    model.save(name)


def wrtie_file(cost_time, loss, acc):
    data = [cost_time, loss, acc]
    with open('model.txt', 'a', newline='', encoding='utf-8') as f:
        for d in data:
            for i, j in d.items():
                f.writelines((i + "   " + str(j) + "    "))
            f.write("\n")
        f.close()
    print("\n all done")


if __name__ == '__main__':
    ###########################################################################
    # 1. cifar10 data
    #X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(100, 200)  # for test
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(20000,40000)

    # images shape: 32,32,3
    ###########################################################################
    # 2. models list
    models_name = ["LeNet5", "AlexNet", "MyCNN"]
    i_ = 0
    temp, history_model, test_loss_model, test_acc_model, spend_time = [], {}, {}, {}, {}
    models_total = get_model(temp)

    ###########################################################################
    # trian LeNet5  AlexNet
    for m in models_total:

        start = time.time()
        s = models_name[i_]
        history, test_loss, test_acc = compile_mode(m, X_train, y_train, X_valid, y_valid, X_test, y_test)
        history_model[models_name[i_]] = history
        test_loss_model[models_name[i_] + "   loss"] = test_loss
        test_acc_model[models_name[i_] + "   acc"] = test_acc
        save_path = s+".h5"
        # m.save(save_path)
        print(save_path)
        model_save(m, save_path)

        end = time.time()
        spend_time[models_name[i_] + "   time"] = end - start
        i_ += 1

    ###########################################################################
    # ResNet
    K.clear_session()
    ResNet = resnet()
    start = time.time()
    ResNet.fit(X_train, y_train, X_valid, y_valid, batch_size=32, num_epochs=20, verbose=1)
    acc, loss = ResNet.evaluate(X_test, y_test)
    history_model["ResNet"] = history
    test_loss_model["ResNet" + "   loss"] = loss
    test_acc_model["ResNet" + "   acc"] = acc
    end = time.time()
    spend_time["ResNet" + "   time"] = end - start
    # ResNet.save_model()

    print("\n done")
    print(history_model, test_loss_model, test_acc_model, spend_time,sep="\n")
    wrtie_file(spend_time, test_loss_model, test_acc_model)

    ###########################################################################
    # for try
    # start = time.time()
    #
    # history, test_loss, test_acc = compile_mode(LeNet5, X_train, y_train, X_valid, y_valid, X_test, y_test)
    # history_model[models_name[i_]] = history
    # test_loss_model[models_name[i_]] = test_loss
    # test_acc_model[models_name[i_]] = test_acc
    #
    # end = time.time()
    # spend_time[models_name[i_]] = end - start
    # i_ += 1
    # print("\n done")
    # print(history_model, test_loss_model, test_acc_model, spend_time,sep="\n")
    # MyCNN = my_cnn()
    # MyCNN.summary()