# NLP

## 文件内容

kaggle nlp： kaggle 上的Nlp项目

nlp basic： python nlp 处理入门，nltk

NLP with CS 224N： MIT自然语言处理神课

NLP with Harvard CS287： NLP课程

NLP with Pytorch：框架使用

NLP with tensorflow：框架使用

##  学习课程资源总结：

[常见面试题汇总](https://blog.csdn.net/qq_17677907/article/details/86448214)

[nlp迁移学习](https://www.toutiao.com/i6637321233358668292/)  [BERT 的发展和展望](https://mp.weixin.qq.com/s/U_pYc5roODcs_VENDoTbiQ)

source： https://github.com/NLP-LOVE/ML-NLP

[论文学习资源](https://zhuanlan.zhihu.com/p/69358248)

[牛津大学NLP](https://github.com/oxford-cs-deepnlp-2017/lectures)

[FastAi NLP](https://github.com/fastai/course-nlp) --- [配套视频课](https://www.fast.ai/2019/07/08/fastai-nlp/)

# 词嵌入

word embedding： 自然语言是表达意义的系统，词是基本单元，词映射到实数向量的技术即为词嵌入。文本数据表示分为，离散表示和分布式表示

### 离散表示

#### one-hot 

也称独热编码：构造文本分词后的字典，每一个词为一个 binary value， 表示该词的位置为1，其余位置为0，矩阵表示

缺点：

- 语料库增加，产生维度越来越高，稀疏矩阵
- 不能保留词与词之间的 关系信息

#### 词袋模型  

Bag of words model:  句子或文档的语句用一个“袋子“表示，不考虑文法及词的顺序

文档的向量可以直接将各词的词向量表示加和

缺点：

- 词向量化后，不一定是词出现越多，权重越大
- 词与词之间没有顺序关系

TF-IDF

term frequency–inverse document frequency : **字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。一个词语在一篇文章中出现次数越多, 同时在所有文档中出现次数越少, 越能够代表该文章**

- TF 词频 Term Frequency
- IDF 逆文本频率指数 Inverse Document Frequency

$$TF_w =  \frac{某一类词条中w出现的次数}{该类中所有的词条数目}$$  $$ IDF=log(\frac{语料库文档总数}{包含词条w的文档总数+1}) $$

$$ TF-IDF = TF * IDF$$

缺点： 

- 词与词之间的关系顺序没有表示

#### n-gram 模型

保持词的顺序，做了一个滑窗的操作，这里的n表示的就是滑窗的大小，例如2-gram模型，也就是把2个词当做一组来处理，然后向后移动一个词的长度，再次组成另一组词，把这些生成一个字典，按照词袋模型的方式进行编码得到结果。改模型考虑了词的顺序。

缺点：

- 随着 n 的大小增加，此表会成指数膨胀

离散表示问题：

- 一般问题可以使用，精度要求较高的场景不大合适
- 无法衡量词与词向量之间的关系
- 文本稀疏问题，n-gram语料库的指数型增长问题

### 分布式表示

用一个词附近的其它词来表示该词，这种方法是基于人的语言表达，认为一个词是由这个词的周边词汇一起来构成精确的语义信息。就好比，物以类聚人以群分，如果你想了解一个人，可以通过他周围的人进行了解，因为周围人都有一些共同点才能聚集起来。

#### 共现矩阵

共同出现的意思，词文档的共现矩阵主要用于发现主题(topic)，用于主题模型

问题：

- 存储字典的空间小号非常大，面临稀疏性问题
- 新增加语料，稳定性发生变化

#### 神经网络表示

##### NNLM： Neural Network Language model)

##### word2vec

- CBOW: Continues Bag of Words 连续词袋模型：获得中间词两边的的上下文，然后用周围的词去预测中间的词
- Skip-gram: 当前词来预测窗口中上下文词出现的概率模型
  - 优化方法： 使用[哈夫曼树(Huffman Tree)]([https://zh.wikipedia.org/wiki/%E9%9C%8D%E5%A4%AB%E6%9B%BC%E7%BC%96%E7%A0%81](https://zh.wikipedia.org/wiki/霍夫曼编码))
  - 负例采样Negative Sampling：

问题：

- 对每个local context window单独训练，没有利用包 含在global co-currence矩阵中的统计信息。
- 对多义词无法很好的表示和处理，因为使用了唯一的词向量

总结：

精度要求不高或者练习可使用离散表示，进行操作

# 子词嵌入

## fasttext

**“dog”和“dogs”分别⽤两个不同的向量表⽰，而模型中并未直接表达这两个向量之间的关系。鉴于此，fastText提出了⼦词嵌⼊(subword embedding)的⽅法，从而试图将构词信息引⼊word2vec中的CBOW**

也可理解为词性还原问题

模型类似CBOW

![fasttext](https://camo.githubusercontent.com/51ad9d2b843713518f8bf38ec910b0021af78011/68747470733a2f2f67697465652e636f6d2f6b6b7765697368652f696d616765732f7261772f6d61737465722f4d4c2f323031392d382d32315f32302d33312d32322e6a706567)

**fastText就不一样了，它是用单词的embedding叠加获得的文档向量，词向量的重要特点就是向量的距离可以用来衡量单词间的语义相似程度**

[FASTTEXT实现](http://www.52nlp.cn/fasttext)

# 全局词嵌入 Glove

**GloVe的全称 Global Vectors for Word Representation** 基于全局词频统计（count-based & overall statistics）的词表征（word representation）工具，它可以把一个

实现步骤：

- 构建共现矩阵
- 词向量和共现矩阵的近似关系
- 构造损失函数
- 训练 GloVe模型

LSA（Latent Semantic Analysis）是一种比较早的count-based的词向量表征工具，它也是基于co-occurance matrix的，只不过采用了基于奇异值分解（SVD）的矩阵分解技术对大矩阵进行降维，而我们知道SVD的复杂度是很高的，所以它的计算代价比较大。还有一点是它对所有单词的统计权重都是一致的。而这些缺点在GloVe中被一一克服了。

# textRNN  textCNN



## textRNN

**textRNN指的是利用RNN循环神经网络解决文本分类问题**，文本分类是自然语言处理的一个基本任务，试图推断出给定文本(句子、文档等)的标签或标签集合。

首先我们需要对文本进行分词，然后指定一个序列长度n（大于n的截断，小于n的填充），并使用词嵌入得到每个词固定维度的向量表示。对于每一个输入文本/序列，我们可以在RNN的每一个时间步长上输入文本中一个单词的向量表示，计算当前时间步长上的隐藏状态，然后用于当前时间步骤的输出以及传递给下一个时间步长并和下一个单词的词向量一起作为RNN单元输入，然后再计算下一个时间步长上RNN的隐藏状态，以此重复...直到处理完输入文本中的每一个单词，由于输入文本的长度为n，所以要经历n个时间步长。

- **流程**1：embedding--->BiLSTM--->concat final output/average all output----->softmax layer

  ![1](https://img-blog.csdnimg.cn/2019030220345678.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NkdV9oYW8=,size_16,color_FFFFFF,t_70)

- 流程2：embedding-->BiLSTM---->(dropout)-->concat ouput--->UniLSTM--->(droput)-->softmax layer

  ![2](https://img-blog.csdnimg.cn/20190302204559873.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NkdV9oYW8=,size_16,color_FFFFFF,t_70)

## textCNN

卷积运算， 时序最大池化层，textCNN模型

# seq2seq

当输⼊和输出都是不定⻓序列时，我们可以使⽤编码器—解码器（encoder-decoder）或者seq2seq模型。**序列到序列模型，简称seq2seq模型。这两个模型本质上都⽤到了两个循环神经⽹络，分别叫做编码器和解码器。编码器⽤来分析输⼊序列，解码器⽤来⽣成输出序列。两 个循环神经网络是共同训练的。**

![xulie](https://camo.githubusercontent.com/6e815c3fb89657efc748245e530566a0f9b4f832/68747470733a2f2f67697465652e636f6d2f6b6b7765697368652f696d616765732f7261772f6d61737465722f4d4c2f323031392d382d32395f31312d31302d342e706e67)

# attention 机制

[转载](https://blog.csdn.net/hpulfc/article/details/80448570)

# Transformer

![transformer](https://camo.githubusercontent.com/7c57ce4cffa2d4bb727cf9c25b6ee42e07e75a23/68747470733a2f2f67697465652e636f6d2f6b6b7765697368652f696d616765732f7261772f6d61737465722f4d4c2f323031392d392d32355f32332d342d32342e706e67)

![encoder](https://camo.githubusercontent.com/f80e2010b888dca6e5f0fea726a70fcbce51cab2/68747470733a2f2f67697465652e636f6d2f6b6b7765697368652f696d616765732f7261772f6d61737465722f4d4c2f323031392d392d32355f32332d32352d31342e706e67)

- encoder，包含两层，一个self-attention层和一个前馈神经网络，self-attention能帮助当前节点不仅仅只关注当前的词，从而能获取到上下文的语义。
- decoder也包含encoder提到的两层网络，但是在这两层中间还有一层attention层，帮助当前节点获取到当前需要关注的重点内容。

# BERT

**Bidirectional Encoder Representation from Transformers**



# XLNet

通用的自回归训练方法 

