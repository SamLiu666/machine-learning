https://web.stanford.edu/~jurafsky/slp3/  inference

数据集：https://zhuanlan.zhihu.com/p/145436365?utm_source=wechat_session&utm_medium=social&utm_oi=50414917517312

书籍：https://web.stanford.edu/~jurafsky/slp3/



## 2.1 Regular Expressions

 traditional text processing

**text normalization**: tokenizing, lemmatization（词形还原），stemming（词根）,segmentation（分词）

[常用的正则表达式](https://www.cnblogs.com/zxin/archive/2013/01/26/2877765.HTML)  [正则语法](https://docs.python.org/zh-cn/3/library/re.html)

**Text Normalization 文本规范化**

1. Tokenizing (segmenting) words  分词 [Natural Language Toolkit (NLTK)](http://www.nltk.org)
2. Normalizing word formats  词形式
3. Segmenting sentences  断句text

 text processing： 过程

| 类型 | 算法                                   |
| ---- | -------------------------------------- |
| 分词 | byte-pair encoding, or BPE， MaxMatch. |
| 词根 | The Porter stemmer(written rules)      |
| 断句 | Stanford CoreNLP toolkit： 标点断句    |



最小**编辑距离 edit distance** :measuring how similar two strings are.

## 3 N-gram

$P(X_1,X_2...X_N) = \Pi_{k=1}^{n}P(X_k|X_1^{k-1})$ chain rule-- Markov assumption -- MLE(maximum likeli hood estimation)

评价语言模型：困惑度(perplexity)：越小越好，越不使人困惑 --- 交叉熵

语言模型泛化：没有出现过的短语、词语，不认识的词语（OOV）

平滑处理：smoothing 文本分类

## 4 Naive Bayes Classification

尝试：做一个相关任务

任务：语义分析，垃圾邮件分类。。。

分类词： Input -> bag of words (忽略单词位置)-> output  

方法：平滑处理，忽略掉测试集中出现不在总词汇表中的词，去掉停顿词stopwords

问题：词出现的次数并不代表它一定是重要的~

朴素贝叶斯作为LM，评价方法：precision, recall, F-measures(精确率和召回率的关系)

![image-20200609113433636](C:\Users\liu\AppData\Roaming\Typora\typora-user-images\image-20200609113433636.png)

统计假设检验？

**项目实践**


```shell
数据描述: 每行都是一篇演讲稿，每行的第一个单词指明了这篇演讲稿所属的党派，RED 指共和党，而 BLUE 指民主党。所有单词和符号都已经被转为大写并由空格分隔方便处理。train.txt 有共和党演讲稿和民主党演讲稿各 23 篇，test.txt 有 6 篇共和党演讲稿，12 篇民主党演讲稿。

需统计：
1. 各个单词在各个分类中出现的数量
2. 各个分类中不重复词的数量

朴素贝叶斯训练步骤：
1. 读取训练和测试数据，去除stopwords，处理成词袋，并统计各类样本对应的词频率：此项目为例
   - 统计类别 Blue 和 Red：词个数，词频率，不重复词的个数
2. 训练模型--生成式判别模型，直接输入测试数据，根据最大似然估计，概率大的为对应类别如果是未出现的词，+1 进行平滑处理
   - Blue，Red 概率比较，取值大的为判别的概率。
3. 识别结果100%
```


## 5 Logistic Regression

组成部分：

1. 特征提取
2. 分类函数，sigmoid  softmax
3. 损失函数：交叉熵：$\hat y$ 和 $y$ 区别多少。 取值概率最大，损失函数取反（值最小）
4. 优化方法：随机梯度下降

Generative and Discriminative Classifiers： 朴素贝叶斯在小数据集上效果更好

例子：sentiment classification（情感分类）

参数：条件损失函数最小化，条件概率最大化--找到最好的参数：学习过程

The Stochastic Gradient Descent Algorithm--SGD

mini-batch,, normalization-overfitting: L2-small weights, L1-big weights

**Multinomial logistic regression--softmax**



![2](https://github.com/SamLiu666/machine-learning/blob/master/paper/Speech%20and%20Language%20Processing/pic/2.png)



**逻辑回归-美团点评情感分类**

```shell
数据描述：xlxs文件，包含多种数据，提取评论和评级两类数据，同时将等级划分为0，1两类
数据处理：对语料分词处理，数据集划分为训练集和测试集，没有去除stopwords
模型训练：数据向量化，输入线性回归模型训练
结果分析：精确率，召回率，F1-score, 混淆矩阵
```

## 6 Vector Semantics and Embeddings

词向量的语义，更符合上下文的表示

- 词根和语境，同义词，相似词，关联词，语义结构和角色，暗示

词和向量：基于共现矩阵（词一起出现）

相似度：点积，余弦相似度

共现矩阵：过于稀疏，直觉：如何更有效的表示词向量：

1. TF-IDF： TF-词频，IDF逆文档频率，实质是共现矩阵的加权，同样会有稀疏问题
   - $tf_{t,d}=lg(count(t,d)+1)$
   - $idf_t=lg(\frac{N}{df_t})$
   - $w_{t,d} = tf_{t,d}\times idf_t$

PMI 点相互信息：x y 出现的频率

Analogy： 推论

## 7 NNLM

units+activation funciton

计算图、反向传播、dropout，mini-batch

Feedforward NN - times, like LM， |V| 个词

embedding(1*n_d) ->W(n_d * d_h ) = hidden (1 * d_h)-> W2( |V|  * d_h) -> output (1 * |V|)

embedding： 对所有词的向量嵌入，因此是 d*|V|  （d for each word d dimention）

![3](https://github.com/SamLiu666/machine-learning/blob/master/paper/Speech%20and%20Language%20Processing/pic/3.jpg](https://github.com/SamLiu666/machine-learning/blob/master/paper/Speech and Language Processing/pic/3.jpg))

## 8 Part-of-Speech Tagging

[HMM参考](https://www.cnblogs.com/mantch/p/11203748.html)

词性标注：named entities and information extraction，包含两种算法：

- generative -- Hidden Markov Model (HMM)- sequence model
- discriminative—the Maximum Entropy Markov Model (MEMM)
- recurrent neural network (RNN).

三者各有应用场景，The Penn Treebank Part-of-Speech Tagset，目的是消歧

HMM：基于现在状态对未来状态的预测，

- Markov Assumption: $P(q_i=a|q_1...q_{i-1})=P(q_i=a|q_{i-1})$ 

HMM as decoding: The decoding algorithm for HMMs is the Viterbi algorithm (维比特算法)； beam searcha

deal with unknow words:

MEMM：直接计算 $\hat T = argmax_T P(T|W)$

以上的模型都是从左往右的，考虑文字双向的情况，可以使用conditional random field or CRF （条件随机场）

## 总结 2--8

这一部分的内容从第二章到第八章，包括正则表达式（分词，词形还原等等），语言模型（N-gram、贝叶斯），这些模型的使用都是需要对文本先进行处理--向量化（word embedding, TF-IDF）,引出语义相似度--余弦。在原本传统的处理基础上，对比提出了基于神经网络的语言模型--NNLM（逻辑回归），输入--向量化（embedding） 经过权重计算，得到隐层值，再经过权重和激活函数（relu, tanh,sigmoid..）得到输出层。通过损失函数(交叉熵，均值等)，反向传播算法（adma, sgd, momentum, adgrad等等），优化参数，得到训练好的模型，进行测试分类，测试句子。

其中的语义消歧、多义词、不知道的词--词性标注的解决方法，提高模型性能

词性标注已经涉及到了序列到序列的方面，接下来的内容是序列的神经网络处理，也是NLP方向的重大突破

**传统的LM 基于马尔可夫假设的N-gram和NNLM(滑动窗口)，是否可以此分类？**

## 9 Seq with RNN

数学公式推导：反向误差传播更新权重

unrolled networks VS computation graph

1. 将展开的RNN网络应用到计算图网络长，逐词逐句的处理
2. 长句子序列，分成几个部分，作为一个分离的训练集

RNN 应用：RNNLM：交叉熵损失函数

1. autoregressive generation： 困惑度 评价模型生成文字的效果
2. Sequence Labeling： 词性标注问题， named entity recognition （IOB 编码）， structure prediction（语义树）
3. Viterbi and Conditional Random Fields (CRFs)
4. RNNs for Sequence Classification：序列太长，前面的词对最终分类结果影响小

### Deep Networks: Stacked and Bidirectional RNNs

**1 Stacked RNNs** ： 一个RNN的输出作为另一个RNN的输入，堆叠RNN网络：发现更多的特征表示，类似CV中的多层CNN，训练成本会上升

**2 Bidirectional RNNs** ：双向RNN，两个独立的RNN网络，组合输出表示上下文

### Managing Context in RNNs: LSTMs and GRUs

应对RNN对长序列中前面单词处理不够的问题

**1 Long short-term memory (LSTM) networks**：使用三个门”记忆“信息, forget , add, output  

**2 Gated Recurrent Units**：简化版 gate，reset  ,update

面临的困境： 词典巨大，也会遇到unkbow word

Character-Level Word Embedding: