import keras
"""使用CNN识别MNIST数字，"""
from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical

# 导入MNIST手写字数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


def build_cnn():
    # CNN 一般选32 或者 64， （3*3）
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3), activation="relu",input_shape=(28,28,1)))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation="relu"))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(64,(3,3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    model.summary()
    model.compile(optimizer="rmsprop",
                  loss= "categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_images, train_labels, epochs=5, batch_size=128)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    return test_acc


test_acc = build_cnn()
print("识别精确率：", test_acc)