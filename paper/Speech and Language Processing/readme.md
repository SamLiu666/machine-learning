https://web.stanford.edu/~jurafsky/slp3/  inference

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

## 5 Logistic Regression

