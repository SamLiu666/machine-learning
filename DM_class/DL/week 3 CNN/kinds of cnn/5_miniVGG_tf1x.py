import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pickle
import os
import numpy as np
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from cifar_data import Cifar10


def display_cifar(images, size):
    n= len(images)
    plt.figure()
    plt.gca().set_axis_off()
    im= np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)]) for j in range(size)])
    plt.imshow(im)
    plt.show()


class Layers:
    @staticmethod
    def dense(inputs, output_size, name="dense1", act=None):
        with tf.name_scope(name):
            input_size = int(inputs.get_shape()[1])
            W_init = tf.random.normal([input_size, output_size], mean=0, stddev=0.1, dtype=tf.float32)
            b_init = tf.random.normal([output_size], mean=0, stddev=0.1, dtype=tf.float32)
            W = tf.Variable(W_init, name="W")
            b = tf.Variable(b_init, name="b")
            Wxb = tf.matmul(inputs, W) + b
            if act is None:
                return Wxb
            else:
                return act(Wxb)

    @staticmethod
    def conv2D(inputs, filter_shape, strides=[1, 1, 1, 1], padding="SAME", name="conv1", act=None):
        with tf.name_scope(name):
            W_init = tf.random.normal(filter_shape, mean=0, stddev=0.1, dtype=tf.float32)
            W = tf.Variable(W_init, name="W")
            b_init = tf.random.normal([int(filter_shape[3])], mean=0, stddev=0.1, dtype=tf.float32)
            b = tf.Variable(b_init, name="b")
            Wxb = tf.nn.conv2d(input=inputs, filter=W, strides=strides, padding=padding) + b
            if act is None:
                return Wxb
            else:
                return act(Wxb)

    @staticmethod
    def max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"):
        return tf.nn.max_pool(value=inputs, ksize=ksize, strides=strides, padding=padding)

    @staticmethod
    def dropout(inputs, drop_rate):
        return tf.nn.dropout(inputs, rate=drop_rate)

    @staticmethod
    def batch_norm(inputs, phase_train):
        return tf.layers.batch_normalization(inputs, momentum=0.90, training=phase_train, center=True, scale=True)


class SimpleCNN():
    def __init__(self, width=32, height=32, depth=3, num_classes=4, drop_rate=0.3, batch_size=10,
                 epochs=5, optimizer=tf.train.AdamOptimizer(learning_rate=0.01)):
        self.width = width
        self.height = height
        self.depth = depth
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.tf_graph = tf.Graph()
        self.session = tf.Session(graph=self.tf_graph)
        self.train_op = None
        self.X = None
        self.y = None
        self.drop_rate_holder = None
        self.phase_train = True

    def build(self):
        with self.tf_graph.as_default():
            with tf.name_scope("simple_cnn"):
                self.X = tf.placeholder(shape=[None, self.height, self.width, self.depth], dtype=tf.float32)
                self.y = tf.placeholder(shape=[None, self.num_classes], dtype=tf.int64)
                self.drop_rate_holder = tf.placeholder(dtype=tf.float32)
                self.phase_train = tf.placeholder(dtype=tf.bool)

                conv1 = Layers.conv2D(inputs=self.X, filter_shape=[5, 5, 3, 32], act=tf.nn.relu)
                pool1 = Layers.max_pool(conv1)
                conv2 = Layers.conv2D(inputs=pool1, filter_shape=[5, 5, 32, 64], act=tf.nn.relu)
                pool2 = Layers.max_pool(conv2)

                flat_dim = pool2.get_shape().as_list()[1] * pool2.get_shape().as_list()[2] * \
                           pool2.get_shape().as_list()[3]
                pool2_flat = tf.reshape(pool2, [-1, flat_dim])

                full1 = Layers.dense(inputs=pool2_flat, output_size=1000, act=tf.nn.relu)
                full1_drop = Layers.dropout(inputs=full1, drop_rate=self.drop_rate_holder)
                self.logits = Layers.dense(inputs=full1_drop, output_size=self.num_classes)

            with tf.name_scope("train"):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.logits)
                self.loss = tf.reduce_mean(cross_entropy)
                self.train_op = self.optimizer.minimize(self.loss)

            with tf.name_scope("predict"):
                self.y_pred = tf.argmax(self.logits, 1)
                corrections = tf.equal(self.y_pred, tf.argmax(self.y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(corrections, tf.float32))

            self.session.run(tf.global_variables_initializer())

    def partial_fit(self, X_batch, y_batch):
        self.session.run([self.train_op], feed_dict={self.X: X_batch, self.y: y_batch,
                                                     self.drop_rate_holder: self.drop_rate,
                                                     self.phase_train: True})

    def predict(self, X, y):
        y_pred, acc = self.session.run([self.y_pred, self.accuracy], feed_dict={self.X: X, self.y: y,
                                                                                self.drop_rate_holder: 0,
                                                                                self.phase_train: False})
        return y_pred, acc

    def compute_acc_loss(self, X, y):
        loss, acc = self.session.run([self.loss, self.accuracy],
                                     feed_dict={self.X: X, self.y: y, self.drop_rate_holder: 0,
                                                self.phase_train: False})
        return loss, acc

    def __exit__(self, exc_type, exc_value, traceback):
        self.session.close()


########################################################################
dataset = Cifar10(200, 400)
X_train, y_train, X_valid, y_valid, X_test, y_test = dataset.get_cifar_data()
# display_cifar(X_train, 10)

batch_size= 32
epochs= 5
num_classes= 10
training_size= len(X_train)
iter_per_epoch= int(training_size/batch_size) +1

network= SimpleCNN(batch_size= batch_size, epochs= epochs,
                   num_classes= num_classes, optimizer= tf.train.AdamOptimizer(learning_rate=0.001))


for epoch in range(epochs):
    for i in range(iter_per_epoch):
        if (i+1)*batch_size<=training_size:
            X_batch,y_batch= X_train[i*batch_size : (i+1)*batch_size], y_train[i*batch_size : (i+1)*batch_size]
        else:
            X_batch, y_batch = X_train[i*batch_size:-1], y_train[i*batch_size:-1]
        network.partial_fit(X_batch, y_batch)

        if i % 100==0:
            batch_loss, batch_acc= network.compute_acc_loss(X_batch, y_batch)
            print("Batch loss: {}, batch accuracy: {}".format(batch_loss, batch_acc))

    train_loss, train_acc= network.compute_acc_loss(X_train, y_train)

    val_loss, val_acc= network.compute_acc_loss(X_valid, y_valid)

    print("\nEpoch {}: train loss={}, val loss={}".format(epoch+1, train_loss, val_loss))
    print("######: train acc={}, val acc={}\n".format(train_acc, val_acc))
    #print(train_out, val_out)

print("Finish training and come to testing")
#print(d.test.labels.shape)
y_pred, acc= network.predict(X_test, y_test)
print("Testing accuracy= {}".format(acc))