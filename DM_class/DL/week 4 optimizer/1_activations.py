import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, models, layers
from tensorflow.keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import os, math,json,copy
from scipy.special import erfc
import numpy as np
from keras.utils import plot_model
from tensorflow.keras.callbacks import BaseLogger
import pandas as pd
from keras import backend as K
K.clear_session()  # Some memory clean-up


#######################################################################################
# 2. Define a custom activation function, loss function, and metric¶
def function_plot(z, y, title=None):
    """z: x; y:f(x)"""
    plt.plot(z, y, "b-", linewidth=2)
    plt.plot([-5, 5], [0, 0], 'k-')
    plt.plot([-5, 5], [-1, -1], 'k--')
    plt.plot([0, 0], [-2.2, 3.2], 'k-')
    plt.grid(True)
    plt.title(title, fontsize=14)
    plt.axis([-5, 5, -1.1, 1.1])
    title += ".jpg"
    print(title)
    path = os.path.join(".\pic", title)
    plt.savefig(path)
    plt.show()


def sigmoid(x):
    return 1 / (1+tf.math.exp(-x))
def derivate_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


def tanh(x):
    return ((1-tf.math.exp(-2*x)) / (1 + tf.math.exp(-2*x)))


def elu(x, alpha=1):
    return tf.where(x<0, alpha*(tf.math.exp(x)-1), x)
def elu_derivation(x, alpha=1):
    return tf.where(x<0, alpha*(tf.math.exp(x)), 1)


def relu(x):
    return tf.maximum(0.0, x)
def relu_derivation(x):
    return tf.where(x<0, 0.0, 1.0)


def leaky_relu(x, alpha=0.05):
    return tf.maximum(alpha*x, x)

def leaky_relu_derivation(x, alpha=0.05):
    return tf.where(x<0, alpha, 1)


def selu(z):
    alpha_0_1 = -np.sqrt(2 / np.pi) / (erfc(1 / np.sqrt(2)) * np.exp(1 / 2) - 1)
    scale_0_1 = (1 - erfc(1 / np.sqrt(2)) * np.sqrt(np.e)) * np.sqrt(2 * np.pi) * (
                2 * erfc(np.sqrt(2)) * np.e ** 2 + np.pi * erfc(1 / np.sqrt(2)) ** 2 * np.e - 2 * (2 + np.pi) * erfc(
            1 / np.sqrt(2)) * np.sqrt(np.e) + np.pi + 2) ** (-1 / 2)

    return scale_0_1 * tf.where(z < 0, alpha_0_1 * (tf.exp(z) - 1), z)

def activation_function_plot():
    z = np.linspace(-5,5,200)

    function_plot(z, sigmoid(z), title="sigmoid function")
    function_plot(z, derivate_sigmoid(z), title="derivation of sigmoid")

    function_plot(z, tanh(z), title="tanh function")

    function_plot(z, elu(z), title="elu function")
    function_plot(z, elu_derivation(z), title="elu derivation")

    function_plot(z, relu(z), title="relu function")
    function_plot(z, relu_derivation(z), title="relu derivation")

    function_plot(z, leaky_relu(z), title="leaky_relu")
    function_plot(z, leaky_relu_derivation(z), title="leakuy_relu derivation")

    function_plot(z, selu(z), title="selu function")

# activation_function_plot()
class my_CE_loss(keras.losses.Loss):
    def __init__(self, eps=1E-10, num_classes=10, **kwargs):
        self.eps = eps
        self.num_classes = num_classes
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        loss = tf.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return loss

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.eps}


# 2.3. Define a custom weight initializer¶
class my_Xavier_initializer(tf.keras.initializers.Initializer):
    def __init__(self, random_seed=4):
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

    def __call__(self, shape, dtype=tf.float32):
        stddev = tf.sqrt(2./(shape[0] + shape[1]))
        return tf.random.normal(shape, stddev=stddev, dtype=dtype)


class my_He_initializer(tf.keras.initializers.Initializer):
    def __init__(self, random_seed=42, alpha=math.sqrt(2)):
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        self.alpha = alpha

    def __call__(self, shape, dtype=tf.float32):
        stddev = self.alpha * tf.sqrt(2. / (shape[0] + shape[1]))
        return tf.random.normal(shape, stddev=stddev, dtype=dtype)


# 2.4. Define a custom metric
class TopkAcc(tf.keras.metrics.Metric):
    def __init__(self, k=5, **kwargs):
        super().__init__(**kwargs)  # handles base args (e.g., dtype)
        self.k = k
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.reshape(y_true, shape=[-1])
        b_array = tf.math.in_top_k(y_true, y_pred, self.k)
        num_corrects = tf.reduce_sum(tf.cast(b_array, tf.float32))
        self.total.assign_add(tf.reduce_sum(num_corrects))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.total / self.count

#######################################################################################
# 3. create MiniVGG network
def create_vgg_model(n_classes=10):
    vgg_model = models.Sequential()
    vgg_model.add(layers.Conv2D(32, (3,3), padding="same", activation="elu", input_shape=(32,32,3)))
    vgg_model.add(layers.BatchNormalization(momentum=0.9))

    vgg_model.add(layers.Conv2D(32, (3,3), padding='same', activation='elu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.01), kernel_initializer= my_He_initializer()))
    vgg_model.add(layers.BatchNormalization(momentum=0.9))
    vgg_model.add(layers.MaxPool2D(pool_size=(2,2)))  #downscale the image size by 2
    vgg_model.add(layers.Dropout(rate=0.25))  # deactivate 25% of neurons for each feed-forward

    vgg_model.add(
        layers.Conv2D(64, (3, 3), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.01),
                      kernel_initializer=my_Xavier_initializer()))
    vgg_model.add(layers.BatchNormalization(momentum=0.9))
    vgg_model.add(
        layers.Conv2D(64, (3, 3), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.01),
                      kernel_initializer=tf.keras.initializers.lecun_normal()))
    vgg_model.add(layers.BatchNormalization(momentum=0.9))
    vgg_model.add(layers.MaxPool2D(pool_size=(2, 2)))  # downscale the image size by 2
    vgg_model.add(layers.Dropout(rate=0.25))  # deactivate 25% of neurons for each feed-forward

    vgg_model.add(layers.Flatten())
    vgg_model.add(layers.Dense(512, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.01), kernel_initializer= 'lecun_normal'))
    vgg_model.add(layers.BatchNormalization(momentum=0.9))
    vgg_model.add(layers.Dropout(rate=0.5))
    vgg_model.add(layers.Dense(n_classes, activation='softmax')) #ten classes in Cifar10
    return vgg_model

def plot_acc(history, title=None):
    plt.plot(history.epoch, history["acc"], "o-")
    plt.axis([0, 30 - 1, -0.0001, 0.012])
    plt.xlabel("accuracy")
    plt.ylabel("Learning Rate")
    plt.title(title, fontsize=14)
    plt.grid(True)
    plt.show()


#######################################################################################
# 1. download and prepare data
K.clear_session()  # Some memory clean-up
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
X_train, X_test = X_train/255.0, X_test/255.0
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# shuffle the dataset
idx = np.arange(X_train.shape[0])
np.random.shuffle(idx)
X_trian = X_train[idx]
y_train = y_train[idx]

X_train, X_valid = X_train[0:5000], X_train[5001:10001]
y_train, y_valid = y_train[0:5000], y_train[5001:10001]
print(X_train.shape, X_valid.shape)
print(y_train.shape, y_valid.shape)


#######################################################################################
# 2.trian model
vgg_model = create_vgg_model()
vgg_model.summary()

# save model structure
#plot_model(vgg_model, to_file='pic/vgg_model.jpg', show_shapes=True, show_layer_names=False)
def basic_train():
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    vgg_model.compile(optimizer= opt, loss= 'sparse_categorical_crossentropy', metrics=['accuracy',TopkAcc(k=5)])
    history1 = vgg_model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=32, epochs=5)
    # plot_acc(history1)
    return history1


#######################################################################################
# 4. Learning Rate Scheduler
# use decay learning rate
def step_decay(epoch, learning_rate):
    # initialize
    init_lr = 0.01
    factor = 0.25
    drop_every = 5
    learning_rate = init_lr * (factor ** (np.float(epoch) / drop_every))
    return learning_rate

def adapt_learning_rate():
    #callbacks = [MyLearningRateScheduler(schedule=step_decay, verbose=1)]
    lr_scheduler = keras.callbacks.LearningRateScheduler(step_decay)
    my_vgg = create_vgg_model(n_classes=10) #create the model
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    my_vgg.compile(optimizer= opt, loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])
    history = my_vgg.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size = 32, epochs =30, callbacks= [lr_scheduler], verbose=1)
    # plot_acc(history, "Exponential Scheduling")
    plt.plot(history.epoch, history.history["lr"], "o-")
    plt.axis([0, 30 - 1, -0.0001, 0.012])
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Exponential Scheduling", fontsize=14)
    plt.grid(True)
    plt.show()


adapt_learning_rate()
#######################################################################################
# 5. Underfitting Overfitting
def save_to_file(H=None, fig_path=None):
    pd.DataFrame(H).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 2) # set the vertical range to [0-1]
    plt.savefig(fig_path)
    plt.close()


class TrainingMonitor(BaseLogger):
    def __init__(self, fig_path=None, json_path=None, model_path=None, model=None):
        super(TrainingMonitor, self).__init__()
        self.fig_path = fig_path
        self.json_path = json_path
        self.model_path = model_path
        self.model = model

    def on_train_begin(self, logs={}):  # triggered when the training gets started
        self.H = {}  # initialize the history dictionary
        if self.json_path is not None:  # if the JSON history path exists, load the training history
            if os.path.exists(self.json_path):
                self.H = json.loads(open(self.json_path).read())

    def on_train_end(self, logs={}):  # triggered when the training gets ended
        if self.model_path != None:
            self.model.save(self.model_path)  # save the current model when finishing the training

    def on_epoch_end(self, epoch, logs={}):
        for (k, v) in logs.items():  # loop over the logs and update the loss, accuracy, etc.
            l = self.H.get(k, [])
            l.append(float(str(v)))
            self.H[k] = l
        # check to see if the training history should be serialized to file
        if self.json_path is not None:
            f = open(self.json_path, "w")
            f.write(json.dumps(self.H))
            f.close()
        if len(self.H["loss"]) > 1:
            save_to_file(self.H, self.fig_path)

model_path = "./logs/mini_vgg.h5"
fig_path = "./logs/history.png"
json_path = "./logs/history.json"

def run_my_vgg(epochs = 10):
    opt = keras.optimizers.Adam(learning_rate=0.001)
    if not os.path.exists(model_path):  #the first time training
        my_vgg = create_vgg_model(n_classes=10) #create the model
        my_vgg.compile(optimizer= opt, loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])
        monitor = TrainingMonitor(fig_path=fig_path, json_path=json_path, model_path=model_path, model= my_vgg)
        callbacks = [monitor]
        my_vgg.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size = 32,
                   epochs =epochs, callbacks= callbacks, verbose=1)
    else:
        # Recreate the exact same model, including its weights and the optimizer
        my_vgg = tf.keras.models.load_model(model_path)
        my_vgg.compile(optimizer= opt, loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])
        monitor = TrainingMonitor(fig_path=fig_path, json_path=json_path, model_path=model_path, model= my_vgg)
        callbacks = [monitor]
        my_vgg.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size = 32,
                   epochs =epochs, callbacks= callbacks, verbose=1)


run_my_vgg(epochs=20)
#######################################################################################
# early stopping
from keras.callbacks import EarlyStopping
early_checkpoint = EarlyStopping(patience=2, monitor='val_loss', mode='min')
callbacks = [early_checkpoint]
opt = keras.optimizers.Adam(learning_rate=0.005)
vgg_model = create_vgg_model()
vgg_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
vgg_model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=32, epochs=20, callbacks=callbacks, verbose=1)

#######################################################################################
# 7. Checkpointing Neural Network Model Improvements