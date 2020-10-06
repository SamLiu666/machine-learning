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
# 2，处理数据
eng, chi = bulid_text(data)
input_tensor, inp_lang = tokenize(eng)
target_tensor, targ_lang = tokenize(chi)

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
#print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))
# 尝试实验不同大小的数据集
num_examples = 30000
# 计算目标张量的最大长度 （max_length）
max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)
# test_= convert_word_tensor(target_tokenizer, "你好 我 是 飞天 怪物")
# print(test_, type(test_))

# 3创建 tf.data数据集
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
print(example_input_batch.shape, example_target_batch.shape)


print("##################################\n ",
      "vocab_inp_size, embedding_dim, units, BATCH_SIZE, vocab_tar_size",
      vocab_inp_size, embedding_dim, units, BATCH_SIZE, vocab_tar_size)


# 找出输入对应的翻译
def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = eng_preprocess_sent(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # 存储注意力权重以便后面制图
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # 预测的 ID 被输送回模型
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot

# 注意力权重制图函数
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

# 翻译
def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))


checkpoint_dir = r'E:\chrome download\paper\training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
print(checkpoint_prefix)

print("#################### load weights ")
optimizer = tf.keras.optimizers.Adam()
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
# 恢复检查点目录 （checkpoint_dir） 中最新的检查点
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

while True:
    str = input("Enter Your Sentence: ")
    translate(str)

    q = input("Stop Enter q")
    if q == 'q':
        break
