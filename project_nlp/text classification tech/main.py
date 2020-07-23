import tensorflow as tf
from keras.datasets import imdb
from tensorflow import keras
import numpy as np

################ for stack
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Restrict TensorFlow to only use the fourth GPU
#         tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print("physical_devices-------------", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.experimental.set_visible_devices([], 'GPU')
##########################

data = keras.datasets.imdb

#「只考虑前 10,000 个最常用的词，但排除前 20 个最常见的词」。
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)
# print(train_data.shape, len(train_labels), len(test_labels))  # 已经是处理好的数字

word_index = imdb.get_word_index()  # 对应字典
word_index = {k:(v+3) for k,v in word_index.items()}
word_index['<PAD>'] = 0
word_index["<START>"] = 1
word_index["UNK"] = 2
word_index["UNUSED"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# 输入数据整理成同样长度
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index['<PAD>'], padding="post", maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index['<PAD>'], padding="post", maxlen=256)

def decode_review(text):
    # 数字转单词
    return " ".join([reverse_word_index.get(i, "?")for i in text])

# print(decode_review(train_data[0]),'\n',train_labels[0:3])
# print(len(test_data[0]), len(test_data[1]))  #  test the length

# 模型训练
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 创造验证集,validation data
x_val = train_data[:10000]
x_train= train_data[10000:]

y_val = train_labels[:10000]
y_train= train_labels[10000:]

history = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

# 测试，评价模型
results = model.evaluate(test_data, test_labels)
print(results)

# 实例检测
test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction:  " + str(predict[0]))
print("Actual:  " + str(test_labels[0]))

# 存储模型
model.save("classify.h5")

# 调用模型
# model = keras.models.load_model("classify.h5")
# def review_encode(s):
#     encode = [1]
#
#     for word in s:
#         if word.lower() in word_index:
#             encode.append(word_index[word.lower()])
#         else:
#             encode.append(2)
#     return encode
#
#
# with open("test.txt", encoding="utf-8") as file:
#     for line in file.readlines():
#         nline = line.replace(",", "").replace(".", "").replace(")", "").replace(":", "").replace("\"", "").split()
#         encode = review_encode(nline)
#
#         encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index['<PAD>'], padding="post", maxlen=256)
#         predict = model.predict(encode)
#         print(line)
#         print(encode)
#         print(predict[0])
