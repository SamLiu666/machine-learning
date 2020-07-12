"""26 letters distinguish NN"""
import os, math
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# running in tfx1
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt


# 1. data processing
data_path = "letter_scale.libsvm"
data_file = os.path.abspath(".\Data\\"+data_path)
# print(data_file)
x, y = load_svmlight_file(data_file)
print(x.shape, y.shape)
x = x.toarray()
y = y.reshape(y.shape[0], -1)
print("data name:x y and size",x.shape, y.shape)
print("classes: {}".format(len(np.unique(y))))

print("############################ category to number")
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
print(y[:4])
print("lables: {}".format(np.unique(y)))


# split the data with sklearn
def data_split(data, target, train_size, test_size):
    print("############################ 3 split the dataset")
    valid = 1 - (train_size+test_size)
    x1, x_test, y1, y_test = train_test_split(data, target, test_size=test_size, random_state=33)
    x_train, x_valid, y_train, y_valid = train_test_split(x1, y1, test_size=float(valid)/(valid+train_size))
    return x_train, x_valid, x_test, y_train, y_valid, y_test


X_train, X_valid, X_test, y_train, y_valid, y_test = data_split(x, y, train_size=0.8, test_size=0.1)
y_train= y_train.reshape(-1)
y_test= y_test.reshape(-1)
y_valid= y_valid.reshape(-1)
print(X_train.shape, X_valid.shape, X_test.shape)
print(y_train.shape, y_valid.shape, y_test.shape)
print(y_train.shape[0]/y_valid.shape[0], y_train.shape[0]/y_test.shape[0])

print("############################ 4 construction")
train_size= int(X_train.shape[0])
n_features= int(X_train.shape[1])
n_classes= len(np.unique(y_train))
n_in= n_features    # dimension of input
n1= 10              # number of hidden units at the first layer
n2= 20              # number of hidden units at the second layer
n3= 15              # number of hidden units at the third layer
n_out= n_classes    # number of classification classes


def dense_layer(inputs, output_size, act=None, name="hidden-layer"):
    print("############################ 4.1 build hidden layer")
    # innitialize parameters
    input_size = int(inputs.get_shape()[1])  #int(inputs.shape[1])
    W_init = tf.random.normal([input_size, output_size], mean=0, stddev=0.1, dtype=tf.float32)
    b_init = tf.random.normal([output_size], mean=0, stddev=0.1,dtype=tf.float32)
    W = tf.Variable(W_init, name="W")
    b = tf.Variable(b_init, name="b")
    Wxb = tf.matmul(inputs, W) + b
    print("{} finished results:{}".format(name, Wxb))
    if act is None:
        return Wxb
    else:
        return act(Wxb)


tf.reset_default_graph()
print("############################ 4.2 train the model")
with tf.name_scope("network"):
    X= tf.placeholder(shape=[None, n_in], dtype= tf.float32)
    y= tf.placeholder(shape=[None], dtype= tf.int32)
    h1= dense_layer(X, n1, act= tf.nn.relu, name= "layer1")
    h2= dense_layer(h1, n2, act= tf.nn.relu, name= "layer2")
    h3= dense_layer(h2, n3, act= tf.nn.relu, name= "layer3")
    logits= dense_layer(h3, n_out, name="logits")

with tf.name_scope("train"):
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits, name='entropy')
    loss = tf.reduce_mean(entropy, name="loss")
    tf.summary.scalar("loss", loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss)

with tf.name_scope('evaluation'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar("accuracy", accuracy)  #summarize the accuracy

print("############################ 4.3 write summary to the data file")
if(not os.path.exists(r".\logs\train")):
    os.makedirs("./logs/train")
if(not os.path.exists(r".\logs\val")):
    os.makedirs("./logs/val")

merged = tf.summary.merge_all()
train_writer= tf.summary.FileWriter("./logs/train")
valid_writer= tf.summary.FileWriter("./logs/val")

print("############################ 4.4  Execution and Evaluation Phase")
batch_size = 32
history = []
num_epoch = 120
iter_per_epoch= math.ceil(float(train_size)/batch_size)  #number of iterations per epoch

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    sess.run(init)
    for epoch in range(num_epoch):
        for idx_start in range(0, X_train.shape[0], batch_size):
            idx_end = min(X_train.shape[0], idx_start+batch_size)
            X_batch, y_batch = X_train[idx_start:idx_end], y_train[idx_start:idx_end]
            sess.run([train_op], feed_dict={X:X_batch, y:y_batch})
        # compute loss and accuracies
        train_summary, train_loss, train_accuracy= sess.run([merged, loss, accuracy], feed_dict={X:X_train, y:y_train})
        train_writer.add_summary(train_summary, epoch + 1)
        train_writer.flush()

        valid_summary, val_loss, val_accuracy = sess.run([merged, loss, accuracy], feed_dict={X: X_valid, y: y_valid})
        valid_writer.add_summary(valid_summary, epoch + 1)
        valid_writer.flush()

        # print and save the results
        print("Epoch {}: valid loss={:.4f}, valid acc={:.4f}".format(epoch + 1, val_loss, val_accuracy))
        print("########: train loss={:.4f}, train acc={:.4f}".format(train_loss, train_accuracy))
        hist_item = {"train_loss": train_loss, "train_acc": train_accuracy,
                     "val_loss": val_loss, "val_acc": val_accuracy}
        history.append(hist_item)

    print("---------------------------------------------\n")
    test_accuracy = sess.run(accuracy, feed_dict={X: X_test, y: y_test})
    print("Test accuracy: {:.4f}".format(test_accuracy))

    print("---------------------------------------------\n")
    print("Save Model")
    if not os.path.exists(os.path.abspath("./models/tmp")):
        os.makedirs(os.path.abspath("./models/tmp"))

    # Save the variables to disk.
    save_path = saver.save(sess, "./models/tmp/model.ckpt")

    # saver.save(sess, './checkpoint_dir/MyModel')
    print("Model saved in path: %s" % save_path)


print("############################ 5  Visualization")
plt.rcParams["figure.figsize"] = (8,10)


def plot_history(history):
    train_losses =[]
    train_accuracies=[]
    valid_losses=[]
    valid_accuracies=[]
    for h_item in history:
        train_losses.append(h_item["train_loss"])
        train_accuracies.append(h_item["train_acc"])
        valid_losses.append(h_item["val_loss"])
        valid_accuracies.append(h_item["val_acc"])
    plt.subplot(2,1,1)
    plt.plot(train_losses, "r.-", label="train loss")
    plt.plot(valid_losses, "b.-", label= "valid loss")
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(train_accuracies, "r.-", label="train acc")
    plt.plot(valid_accuracies, "b.-", label= "valid acc")
    plt.legend()

    plt.show()

plot_history(history)

print("############################ 6  Resotre Model not done")
# meta_path = r'./save_models/lstm-attention.ckpt.meta'
# ckpt_path = './save_models/lstm-attention.ckpt'
meta_path = r'D:\machine learning\DM_class\DL\week 2\models\tmp\model.ckpt.meta'
ckpt_path = r"D:\machine learning\DM_class\DL\week 2\models\tmp\model.ckpt"
# tf.reset_default_graph()
#
# with tf.Session() as sess:
#     # 1. 加载graph
#     saver=tf.train.import_meta_graph(meta_path)
#     saver.restore(sess, ckpt_path)
#     graph = tf.get_default_graph()
#
#     # 2. graph.get_operation_by_name获取需要feed的placeholder
#     # 注意: 这些tensor的名字需要跟模型创建的时候对应
#     # x = graph.get_operation_by_name('inputX').outputs[0]
#     # y = graph.get_operation_by_name('inputY').outputs[0]
#     # dropout1 = graph.get_operation_by_name('dropoutKeepProb').outputs[0]
#     # dropout2 = graph.get_operation_by_name('denseKeepProb').outputs[0]
#     feed_dict = {X: X_test, y: y_test}
#
#     # 3. tf.get_collection获取预测结果
#     # 注意: 在训练代码中，需要计算的tensor要先用tf.add_to_collection
#     tf.add_to_collection('network', logits)
#     p = tf.get_collection('network')[0]
#
#     # 4. sess.run获得模型的预测输出
#     prediction = sess.run([p], feed_dict=feed_dict)
#     print(prediction[:20])

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
# with tf.Session() as sess:
#     # Restore variables from disk.
#     # saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel.meta')
#     saver = tf.train.import_meta_graph(r"D:\machine learning\DM_class\DL\week 2\models\tmp\model.ckpt.meta")
#     saver.restore(sess, "./models/tmp/model.ckpt")
#     print("Model restored.")
#
#     # present the parameters
#     tvs = [v for v in tf.trainable_variables()]
#     print("All the paramters: ")
#     for v in tvs:
#         print(v.name)
#     pred = sess.run(output, feed_dict={X_test: res_image})
#     print(np.argmax(pred, 1))