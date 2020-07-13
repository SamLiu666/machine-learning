import os
import numpy as np
from sklearn.datasets import load_svmlight_file


data_file_name= "D:\东蒙 人工智能课程\DM_class\FIT5215 Deep Learning\session 2\Tute 2\Data\letter_scale.libsvm"
data_file = os.path.abspath(data_file_name)
X_data, y_data = load_svmlight_file(data_file)
X_data= X_data.toarray()
y_data= y_data.reshape(y_data.shape[0],-1)
print("X data shape: {}".format(X_data.shape))
print("y data shape: {}".format(y_data.shape))
print("# classes: {}".format(len(np.unique(y_data))))
print(np.unique(y_data))
print("---------------------------------------------\n")

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def train_valid_test_split(data, target, train_size, test_size):
    valid_size = 1 - (train_size + test_size)
    X1, X_test, y1, y_test = train_test_split(data, target, test_size = test_size, random_state= 33)
    X_train, X_valid, y_train, y_valid = train_test_split(X1, y1, test_size = float(valid_size)/(valid_size+ train_size))
    return X_train, X_valid, X_test, y_train, y_valid, y_test

le = preprocessing.LabelEncoder()
le.fit(y_data.ravel())
y_data= le.transform(y_data)
print(y_data[:])

X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(X_data, y_data, train_size=0.8, test_size=0.1)
y_train= y_train.reshape(-1)
y_test= y_test.reshape(-1)
y_valid= y_valid.reshape(-1)
print(X_train.shape, X_valid.shape, X_test.shape)
print(y_train.shape, y_valid.shape, y_test.shape)
print("lables: {}".format(np.unique(y_train)))

print("---------------------------------------------\n")
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
print(tf.__version__)
print(keras.__version__)


train_size= int(X_train.shape[0])
n_features= int(X_train.shape[1])
n_classes= len(np.unique(y_train))
model = Sequential()
model.add(Dense(units=n_features, input_shape=(16, ), activation="relu"))
model.add(Dense(units=10, activation="relu"))
model.add(Dense(units=20, activation="relu"))
model.add(Dense(units=15, activation="relu"))
model.add(Dense(units=n_classes, activation="softmax"))

model.build()
model.summary()

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x=X_train, y=y_train, batch_size=32, epochs=100, validation_data=(X_valid, y_valid))

model.evaluate(X_test, y_test)


X_new = np.reshape(X_test[10, :], (1,-1))
y_prob = model.predict(X_new)
y_pred = model.predict_classes(X_new)
if y_pred[0]==y_test[10]:
    print("Corrected predeiction !")
else:
    print("Incorrected prediction !")

import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()

optimizer_names = ["Nadam", "Adam", "Adadelta", "Adagrad", "RMSprop", "SGD"]
optimizer_list = [keras.optimizers.Nadam(learning_rate=0.001), keras.optimizers.Adam(learning_rate=0.001),
                  keras.optimizers.Adadelta(learning_rate=0.001),
                  keras.optimizers.Adagrad(learning_rate=0.001), keras.optimizers.RMSprop(learning_rate=0.001),
                  keras.optimizers.SGD(learning_rate=0.001)]
best_acc = 0
best_i = -1
for i in range(len(optimizer_list)):
    print("*Evaluating with {}\n".format(str(optimizer_names[i])))
    model.compile(optimizer=optimizer_list[i], loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=X_train, y=y_train, batch_size=32, epochs=30, validation_data=(X_valid, y_valid), verbose=0)
    acc = model.evaluate(X_test, y_test)[1]
    print("The test accuracy is {}\n".format(acc))
    if acc > best_acc:
        best_acc = acc
        best_i = i
print("The best accuracy is {} with {}".format(best_acc, optimizer_names[best_i]))
