import keras
from keras.layers import Embedding
from keras import backend as K
import os
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import time
start = time.time()

os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0'
K.clear_session()

def use_word_embedding():
    embedding_layer = Embedding(1000, 64)
    max_features = 10000
    maxlen = 20
    # Load the data as lists of integers.
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    # print(x_train)
    # 整数 -> 2D tensor
    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
    # print(x_train)

    # 构建模型
    model = Sequential()
    model.add(Embedding(10000, 8, input_length=maxlen))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    history = model.fit(x_train, y_train,
                        epochs=10, batch_size=32, validation_split=0.2)


def extract_text(train_dir):
    labels = []
    texts = []

    for label in ['neg', 'pos']:
        dir_name = os.path.join(train_dir, label)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname), 'r', encoding='utf-8')
                texts.append(f.read())
                f.close()
                if label == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)
                print(fname, 'done')
    return texts,labels


imdb_dir = r"E:\chrome download\paper\corpus\aclImdb"
train_dir = os.path.join(imdb_dir, "train")
texts, labels = extract_text(train_dir)

# 数据tokenize
maxlen = 100  # We will cut reviews after 100 words
# training_samples = 200  # We will be training on 200 samples
training_samples = 20000
validation_samples = 5000  # We will be validating on 10000 samples
max_words = 10000  # We will only consider the top 10,000 words in the dataset

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# Split the data into a training set and a validation set
# But first, shuffle the data, since we started from data
# where sample are ordered (all negative first, then all positive).
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

# 导入预训练好的Glove 模型
glove_dir = r'E:\chrome download\paper\corpus\glove.6B'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), 'r',encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# embedding layer
embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

# build model
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Load the GloVe embeddings in the mode
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

# trian and evaluate
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=11,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')

# visualization
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

K.clear_session()
# test prediction
imdb_dir = r"E:\chrome download\paper\corpus\aclImdb\test"
texts, labels = extract_text(imdb_dir)
sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)
model.load_weights('pre_trained_glove_model.h5')
print(model.evaluate(x_test, y_test))

end = time.time()
print("time spend: ", end-start)