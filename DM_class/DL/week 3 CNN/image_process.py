import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import cv2
import os
from imutils import paths
import imutils
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        image = cv2.resize(image, (self.width, self.height), interpolation=self.inter)
        b, g, r = cv2.split(image)
        image = cv2.merge((r, g, b))
        return image


class AnimalsDatasetManager:
    def __init__(self, preprocessors=None):
        self.cur_pos = 0
        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors = list()

    def load(self, label_folder_dict, max_num_images=500, verbose=-1):
        data = list();
        labels = list()
        for label, folder in label_folder_dict.items():
            image_paths = list(paths.list_images(folder))
            print(label, len(image_paths))
            for (i, image_path) in enumerate(image_paths):
                image = cv2.imread(image_path)
                if self.preprocessors is not None:
                    for p in self.preprocessors:
                        image = p.preprocess(image)
                data.append(image);
                labels.append(label)
                if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                    print("Processed {}/{}".format(i + 1, max_num_images))
                if i + 1 >= max_num_images:
                    break

        self.data = np.array(data)
        self.labels = np.array(labels)
        self.train_size = int(self.data.shape[0])

    def process_data_label(self):
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(self.labels)
        self.labels = label_encoder.transform(self.labels)
        self.data = self.data.astype("float") / 255.0
        self.classes = label_encoder.classes_

    def train_valid_test_split(self, train_size=0.8, test_size=0.1, rand_seed=33):
        valid_size = 1 - (train_size + test_size)
        X1, X_test, y1, y_test = train_test_split(self.data, self.labels, test_size=test_size, random_state=rand_seed)
        self.X_test = X_test
        self.y_test = y_test
        X_train, X_valid, y_train, y_valid = train_test_split(X1, y1,
                                                              test_size=float(valid_size) / (valid_size + train_size))
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

    def next_batch(self, batch_size=32):
        end_pos = self.cur_pos + batch_size
        x_batch = []
        y_batch = []
        if end_pos <= self.train_size:
            x_batch = self.X_train[self.cur_pos:end_pos, :]
            y_batch = self.y_train[self.cur_pos:end_pos]
            self.cur_pos = end_pos
        else:
            cur_pos_new = (end_pos - 1) % self.train_size + 1
            x_batch = np.concatenate((self.X_train[self.cur_pos: self.train_size, :], self.X_train[0:cur_pos_new, :]))
            y_batch = np.concatenate((self.y_train[self.cur_pos: self.train_size], self.y_train[0:cur_pos_new]))
            self.cur_pos = cur_pos_new
        return x_batch, y_batch


def create_label_folder_dict(adir):
    sub_folders= [folder for folder in os.listdir(adir)
                  if os.path.isdir(os.path.join(adir, folder))]
    label_folder_dict= dict()
    for folder in sub_folders:
        item= {folder: os.path.abspath(os.path.join(adir, folder))}
        label_folder_dict.update(item)
    return label_folder_dict


label_folder_dict= create_label_folder_dict("./Data/Animals")
sp = SimplePreprocessor(width=32, height=32)
data_manager = AnimalsDatasetManager([sp])
data_manager.load(label_folder_dict, verbose=100)
data_manager.process_data_label()
data_manager.train_valid_test_split()
print(data_manager.X_train.shape, data_manager.y_train.shape)
print(data_manager.X_valid.shape, data_manager.y_valid.shape)
print(data_manager.X_test.shape, data_manager.y_test.shape)
print(data_manager.classes)