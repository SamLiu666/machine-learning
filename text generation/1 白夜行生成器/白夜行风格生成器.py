#!/usr/bin/env python
# coding: utf-8

# In[2]:


import jieba
import numpy as np
import sys
import matplotlib.pyplot as plt
import tensorflow as tf


# In[3]:


def sample(p, temperature=1.0):  # 定义采样策略
    distribution = np.log(p) / temperature
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)

p = [0.05, 0.2, 0.1, 0.5, 0.15]

for i, t in zip(range(4), [0.1, 0.4, 0.8, 1.5]):
    plt.subplot(2, 2, i+1)
    plt.bar(np.arange(5), sample(p, t))
    plt.title("temperature = %s" %t, size=16)
    plt.ylim(0,1)


# In[4]:


# 导入数据：白夜行文本
file_path = r"D:\machine learning\text generation\data\白夜行.txt"


# In[5]:


# 分词，构建词典
whole = open(file_path, encoding='utf-8').read()
all_words = list(jieba.cut(whole, cut_all=False))  # jieba分词
words = sorted(list(set(all_words))) # 不重复词
# word_indices = dict((word, index) for word, index in enumerate(words))
word_indices = dict((word, words.index(word)) for word in words)
print("字典单词，字典数量", len(all_words),len(word_indices.values()))


# In[6]:


# 构建序列长度, 30
maxlen = 30
sentences = []
next_word = []

for i in range(0, len(all_words) - maxlen):
    sentences.append(all_words[i: i + maxlen]) # 句子
    next_word.append(all_words[i + maxlen])  # 句子对应的单词
print('提取的句子总数:', len(sentences))


# In[7]:


# 构建词向量
# x: [sentence, seq_length],235804, 30
x = np.zeros((len(sentences), maxlen), dtype='float32') # Embedding的输入是2维张量（句子数，序列长度）
y = np.zeros((len(sentences)), dtype='float32') # (235804,)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        x[i, t] = word_indices[word]
    y[i] = word_indices[next_word[i]]
print(x.shape,y.shape)


# In[8]:


print(np.round((sys.getsizeof(x) / 1024 / 1024 / 1024), 2), "GB") 


# In[9]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras


# In[10]:


main_input = layers.Input(shape=(maxlen, ), dtype='float32') 
model_1 = layers.Embedding(len(words), 128, input_length=maxlen)(main_input)
model_1 = layers.Bidirectional(layers.GRU(256, return_sequences=True))(model_1)
model_1 = layers.Bidirectional(layers.GRU(128))(model_1)
output = layers.Dense(len(words), activation='softmax')(model_1)  
model = keras.models.Model(main_input, output)


# In[11]:


model.summary()


# In[13]:


optimizer = tf.keras.optimizers.RMSprop(lr=3e-3)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)


# In[14]:


model.fit(x, y, epochs=64, batch_size=1024, verbose=2)


# In[15]:


# 文本生成函数
def write_2(model, temperature, word_num, begin_sentence):
    gg = begin_sentence[:30]
    print(''.join(gg), end='/// ')
    for _ in range(word_num):
        sampled = np.zeros((1, maxlen)) 
        for t, char in enumerate(gg):
            sampled[0, t] = word_indices[char]
    
        preds = model.predict(sampled, verbose=0)[0]
        if temperature is None:
            next_word = words[np.argmax(preds)]
        else:
            next_index = sample(preds, temperature)
            next_word = words[next_index]
            
        gg.append(next_word)
        gg = gg[1:]
        sys.stdout.write(next_word)
        sys.stdout.flush()


# In[24]:


begin_sentence = """警察们一齐上前追赶。这里是二楼，桐原正跑向业已停止的扶梯，笹垣相信他已无法脱身。
    但桐原并没有跑上扶梯，而是停下脚步，毫不迟疑地翻身跳往一楼
 """


# In[25]:


# begin_sentence = whole[50003: 50100]
print(begin_sentence[:30])
begin_sentence = list(jieba.cut(begin_sentence, cut_all=False))
#print(begin_sentence, len(begin_sentence))


# In[26]:


write_2(model, None, 300, begin_sentence)


# ## 保存模型

# In[27]:


# Save the model
model.save('path_to_my_model.h5')


# In[28]:


# Recreate the exact same model purely from the file
new_model = keras.models.load_model('path_to_my_model.h5')


# In[29]:


new_model.summary()


# # tensorflow 官方改编

# In[5]:


# 读取并为 py2 compat 解码
path_to_file = r"D:\machine learning\text generation\data\白夜行.txt"
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# 文本长度是指文本中的字符个数
print ('Length of text: {} characters'.format(len(text)))


# In[6]:


# jieba 分词
text = list(jieba.cut(text, cut_all=False))  # jieba分词


# In[8]:


vocab  = sorted(set(text))


# In[9]:


vocab


# ## 处理文本，词典数字对应

# In[18]:


# 创建从非重复字符到索引的映射
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text]) # 将文本转换为数字


# In[17]:


print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')


# In[11]:


# 显示文本首 13 个字符的整数映射
print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))


# ## 创建训练样本和目标

# In[15]:


# 设定每个输入句子长度的最大值
seq_length = 100
examples_per_epoch = len(text)//seq_length

# 创建训练样本 / 目标
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int) # 构建数据集

for i in char_dataset.take(5):
  print(idx2char[i.numpy()])


# In[16]:


sequences = char_dataset.batch(seq_length+1, drop_remainder=True) 

for item in sequences.take(5):
  print(repr(''.join(idx2char[item.numpy()])))


# In[19]:


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)


# In[21]:


for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))
for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))


# ## 创建训练批次

# In[22]:


# 批大小
BATCH_SIZE = 64

# 设定缓冲区大小，以重新排列数据集
# （TF 数据被设计为可以处理可能是无限的序列，
# 所以它不会试图在内存中重新排列整个序列。相反，
# 它维持一个缓冲区，在缓冲区重新排列元素。） 
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset


# In[24]:


# 词集的长度
vocab_size = len(vocab)

# 嵌入的维度
embedding_dim = 256

# RNN 的单元数量
rnn_units = 1024


# In[29]:


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model


# In[30]:


model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)


# In[31]:


for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")


# In[ ]:




