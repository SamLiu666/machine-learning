import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
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
import sys
sys.path.append(r"E:\chrome download\paper\corpus\cmn-eng")
from langconv import *
import tensorflow_datasets as tfds


def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence

def chi_process_sent(w):
    """处理中文句子"""
    w = Traditional2Simplified(w)  # 繁体转中文
    w = jieba.cut(w, cut_all=False)  # 精确模式，避免重复次
    w = " ".join(w)
    w = "<start> " + w + " <end>"
    return w

# 将 unicode 文件转换为 ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def eng_preprocess_sent(w):
    """处理英文句子"""
    # 在单词与跟在其后的标点符号之间插入一个空格
    # 例如： "he is a boy." => "he is a boy ."
    # 参考：https://stackoverflow.com/questions/3645931/peng_ython-padding-punctuation-with-white-spaces-keeping-punctuation
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    # 除了 (a-z, A-Z, ".", "?", "!", ",")，将所有字符替换为空格
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    w = '<start> ' + w + ' <end>'
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
# 2，处理数据
eng, chi = bulid_text(data)
# 查看英文，中文对照
for i in range(20000,20005):
    print("###################### check")
    print(eng[i], " ", chi[i])

    tf_str = tf.constant(eng[i], dtype=tf.string)
    tf_str1 = tf.constant(chi[i], dtype=tf.string)
    print(tf_str,tf_str1)

train_examples = []
for i in range(20000):
    train_examples.append([tf.constant(eng[i], dtype=tf.string),
                           tf.constant(chi[i], dtype=tf.string)])
# 讲数据转换为tf.string

print(train_examples[:4])

print("######################## encoder")
start = time.time()
tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)

tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)
end = time.time()
print("start - end time", start - end)

sample_string = 'Transformer is awesome.'

tokenized_string = tokenizer_en.encode(sample_string)
print ('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer_en.decode(tokenized_string)
print ('The original string: {}'.format(original_string))
print("####### 分割")
sample_string = '他的女儿假装不认识他'
sample_string = chi_process_sent(sample_string)
tokenized_string = tokenizer_en.encode(sample_string)
print ('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer_en.decode(tokenized_string)
print ('The original string: {}'.format(original_string))