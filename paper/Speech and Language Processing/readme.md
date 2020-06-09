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

![image-20200609203105775](C:\Users\liu\AppData\Roaming\Typora\typora-user-images\image-20200609203105775.png)

**逻辑回归-美团点评情感分类**

```shell
数据描述：xlxs文件，包含多种数据，提取评论和评级两类数据，同时将等级划分为0，1两类
数据处理：对语料分词处理，数据集划分为训练集和测试集，没有去除stopwords
模型训练：数据向量化，输入线性回归模型训练
结果分析：精确率，召回率，F1-score, 混淆矩阵
```

