## Learning

## bag of words 

|V|：每个词出现的频率。 词袋模型忽略语法、位置、句子边界等等，只关注单词--**word**

思想:每个词分配一个权重

NaiveBayes: generative model

[好文WordEmebdding--Bert](https://www.jiqizhixin.com/articles/2018-12-10-8):

WordEmbedding: 多义词问题，无论什么词经过模型，预测词的结果始终是同一个

解决方法-ELMO：基于上下文的Embedding，特征抽取器采用的LSTM作为特征选择器

GPT：Generative Pre-Training，特征抽取器采用的transformer