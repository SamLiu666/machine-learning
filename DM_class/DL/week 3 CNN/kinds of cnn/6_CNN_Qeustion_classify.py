from __future__ import print_function
import collections
import math,os,random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import datasets, models, layers
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras import backend as K
K.clear_session()  # Some memory clean-up


class DataManager:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.max_sentence_len = 0
        self.questions = list()
        self.str_labels = list()
        self.numeral_labels = list()
        self.numeral_data = list()
        self.cur_pos = 0

    def maybe_download(self, dir_name, file_name, url):
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        if not os.path.exists(os.path.join(dir_name, file_name)):
            urlretrieve(url + file_name, os.path.join(dir_name, file_name))
        if self.verbose:
            print("Downloaded successfully {}".format(file_name))

    def read_data(self, dir_name, file_name):
        file_path = os.path.join(dir_name, file_name)
        self.questions = list();
        self.labels = list()
        with open(file_path, "r", encoding="latin-1") as f:
            for row in f:
                row_str = row.split(":")
                label, question = row_str[0], row_str[1]
                question = question.lower()
                self.labels.append(label)
                self.questions.append(question.split())
                if self.max_sentence_len < len(self.questions[-1]):
                    self.max_sentence_len = len(self.questions[-1])
        le = preprocessing.LabelEncoder()
        le.fit(self.labels)
        self.numeral_labels = le.transform(self.labels)
        self.str_classes = le.classes_
        self.num_classes = len(self.str_classes)
        if self.verbose:
            print("Sample questions \n")
            print(self.questions[0:5])
            print("Labels {}\n\n".format(self.str_classes))

    def padding(self, length):
        for question in self.questions:
            question = question.extend(["pad"] * (length - len(question)))

    def build_numeral_data(self, dictionary):
        self.numeral_data = list()
        for question in self.questions:
            data = list()
            for word in question:
                data.append(dictionary[word])
            self.numeral_data.append(data)
        if self.verbose:
            print('Sample numeral data \n')
            print(self.numeral_data[0:5])

    def train_valid_split(self, train_size=0.9, rand_seed=33):
        X_train, X_valid, y_train, y_valid = train_test_split(np.array(self.numeral_data),
                                                              np.array(self.numeral_labels),
                                                              test_size=1 - train_size, random_state=rand_seed)
        self.train_numeral = X_train
        self.train_labels = y_train
        self.valid_numeral = X_valid
        self.valid_labels = y_valid

    @staticmethod
    def build_dictionary_count(questions):
        count = []
        dictionary = dict()
        words = []
        for question in questions:
            words.extend(question)
        count.extend(collections.Counter(words).most_common())
        for word, freq in count:
            dictionary[word] = len(dictionary)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return dictionary, reverse_dictionary, count

    def next_batch(self, batch_size, vocab_size, input_len):
        data_batch = np.zeros([batch_size, input_len, vocab_size])
        label_batch = np.zeros([batch_size, self.num_classes])
        train_size = len(self.train_numeral)
        for i in range(batch_size):
            for j in range(input_len):
                data_batch[i, j, self.train_numeral[self.cur_pos][j]] = 1
            label_batch[i, self.train_labels[self.cur_pos]] = 1
            self.cur_pos = (self.cur_pos + 1) % train_size
        return data_batch, label_batch

    def convert_to_feed(self, data_numeral, label_numeral, input_len, vocab_size):
        data2feed = np.zeros([data_numeral.shape[0], input_len, vocab_size])
        label2feed = np.zeros([data_numeral.shape[0], self.num_classes])
        for i in range(data_numeral.shape[0]):
            for j in range(input_len):
                data2feed[i, j, data_numeral[i][j]] = 1
            label2feed[i, label_numeral[i]] = 1
        return data2feed, label2feed


def build_cnn():
    my_model = models.Sequential()
    my_model.add(layers.Conv1D(32, 9, padding="same", activation="relu", input_shape=(38, 8916)))
    my_model.add(layers.BatchNormalization(momentum=0.9))
    my_model.add(layers.Dropout(rate=0.25))

    my_model.add(layers.Conv1D(64, 9, padding="same", activation="relu"))
    my_model.add(layers.BatchNormalization(momentum=0.9))
    my_model.add(layers.MaxPool1D(pool_size=2, padding="same"))
    my_model.add(layers.Dropout(rate=0.25))

    my_model.add(layers.Conv1D(64, 9, padding="same", activation="relu"))
    my_model.add(layers.BatchNormalization(momentum=0.9))
    my_model.add(layers.MaxPool1D(pool_size=2, padding="same"))
    my_model.add(layers.Dropout(rate=0.25))

    my_model.add(layers.Flatten())
    my_model.add(layers.Dense(512, activation="relu"))
    my_model.add(layers.BatchNormalization(momentum=0.9))
    my_model.add((layers.Dropout(rate=0.25)))
    my_model.add(layers.Dense(6, activation="softmax"))
    #my_model.summary()
    return my_model


###################################################################################################################
# data manage
train_dm = DataManager()
test_dm = DataManager()

train_dm.read_data("E:\chrome download\paper\corpus\Question_class", "train_5500.label")
test_dm.read_data("E:\chrome download\paper\corpus\Question_class", "TREC_10.label")
pad_len = max(train_dm.max_sentence_len, test_dm.max_sentence_len)
train_dm.padding(pad_len)
test_dm.padding(pad_len)
all_questions= list(train_dm.questions)
all_questions.extend(test_dm.questions)
dictionary,_,_= DataManager.build_dictionary_count(all_questions)
train_dm.build_numeral_data(dictionary)
test_dm.build_numeral_data(dictionary)
train_dm.train_valid_split()
data_batch, label_batch= train_dm.next_batch(batch_size=5, vocab_size= len(dictionary), input_len= pad_len)
print("Sample data batch- label batch \n")
print(data_batch.shape)
print(label_batch.shape)

###################################################################################################################
cnn_model = build_cnn()

batch_size= 32
epochs= 100
# train_size= len(train_dm.train_numeral)
# iter_per_epoch= math.ceil(train_size/batch_size)
# network= SC_CNN(height= pad_len, width= len(dictionary),batch_size=batch_size, epochs= epochs, num_classes= train_dm.num_classes)
# network.build()

train2feed, train_label2feed= train_dm.convert_to_feed(train_dm.train_numeral, train_dm.train_labels,
                                     input_len= pad_len, vocab_size=(len(dictionary)//4+1))

valid2feed, valid_label2feed=  train_dm.convert_to_feed(train_dm.valid_numeral, train_dm.valid_labels,
                                     input_len= pad_len, vocab_size=len(dictionary))

test2feed, test_label2feed= test_dm.convert_to_feed(np.array(test_dm.numeral_data), np.array(test_dm.numeral_labels),
                                     input_len= pad_len, vocab_size=len(dictionary))

print(train2feed, train2feed.shape)