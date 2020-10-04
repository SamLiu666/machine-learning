import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from my_attention import *

import unicodedata
import re
import numpy as np
import os
import io
import time
# 设置gpu内存自增长
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


print("####################### handle text")
import pandas as pd
import jieba
from langconv import *
def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence


def eng_preprocess_sent(w):
    """处理英文句子"""
    # 在单词与跟在其后的标点符号之间插入一个空格
    # 例如： "he is a boy." => "he is a boy ."
    # 参考：https://stackoverflow.com/questions/3645931/peng_ython-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    # 除了 (a-z, A-Z, ".", "?", "!", ",")，将所有字符替换为空格
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    w = '<start> ' + w + ' <end>'
    return w


def chi_process_sent(w):
    """处理中文句子"""
    w = Traditional2Simplified(w)  # 繁体转中文
    w = jieba.cut(w, cut_all=False)  # 精确模式，避免重复次
    w = " ".join(w)
    w = "<start> " + w + " <end>"
    return w

def bulid_text(data):
    """获取 数据集： 英文，中文"""
    eng, chi = [], []
    for sent in data[0]:
        eng.append(eng_preprocess_sent(sent))
    for sent in data[1]:
        chi.append(chi_process_sent(sent))
    return eng, chi

def tokenize(corpus):
    """令牌化+向量化"""
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(corpus)
    tensor = lang_tokenizer.texts_to_sequences(corpus)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, lang_tokenizer

def max_length(tensor):
    return max(len(t) for t in tensor)

def convert(token, tensor):
    """tensor 转换到词"""
    for t in tensor:
        if t!=0:
          print ("%d ----> %s" % (t, token.index_word[t]))

def convert_word_tensor(token, corpus):
    """tensor 转换到词"""
    tensor = token.texts_to_sequences(corpus)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor

# 1.读取数据
path_to_file = r"E:\chrome download\paper\corpus\cmn-eng\cmn.txt"
data = pd.read_table(path_to_file,header=None)

# 2，处理数据
eng, chi = bulid_text(data)
input_tensor, input_tokenizer = tokenize(eng)
target_tensor, target_tokenizer = tokenize(chi)

# 查看英文，中文对照
for i in range(20000,20005):
    print("###################### check")
    print(eng[i], " ", chi[i])
    print(input_tensor[i], len(input_tensor[i]),"\n",
          target_tensor[i], len(target_tensor[i]), type(target_tensor[i]))

print("input_tensor length:",max_length(input_tensor), "\ntarget_tensor length: ", max_length(target_tensor))

# 采用 80 - 20 的比例切分训练集和验证集
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)
# 显示长度
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))


print ("Input Language; index to word mapping")
convert(input_tokenizer, input_tensor_train[0])
print ()
print ("Target Language; index to word mapping")
convert(target_tokenizer, target_tensor_train[0])

# test_= convert_word_tensor(target_tokenizer, "你好 我 是 飞天 怪物")
# print(test_, type(test_))

# 3创建 tf.data数据集
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(input_tokenizer.word_index)+1
vocab_tar_size = len(input_tokenizer.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
print(example_input_batch.shape, example_target_batch.shape)