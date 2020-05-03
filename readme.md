# 动手学机器学习

此项目用于收集入门级别的机器学习项目，实现机器学习算法。

内容简介：

- 监督学习  Supervised machine learning
  - 决策树 Decision Trees
  - 朴素贝叶斯 Naive Bayes
  - K近邻  K Neaerest Neighbour
  - 回归  Regression
  - 支持向量机 Support Vector Machines
- 聚类 Clustering 无监督学习 Unsupervised machine learning
  - 聚类 clustering problem
  - 相似度量 Similarity measures
  - K均值算法 The K-means algorithm
- 强化学习 reinforcement learning

# 监督学习

## 决策树 Decision Trees

依据分类原则可分为两类：基尼系数，熵

![1](http://www.learnbymarketing.com/wp-content/uploads/2016/02/entropy-formula.png)
![2](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRzYHkcmZKKp2sJN1HpHvw-NgqbD9EnapnbXozXRgajrSGvEnYy&s)

**DT_project** : 根据汽车数据( car_evaluation.csv，采用.OrdinalEncoder 按类编码) 建立的两棵树，进行汽车安全性的预测：

- 两者预测结果相近，可能是数据样本较少
- 训练误差和测试误差接近未出现过拟合情况，如有过拟合情况，建议采取后剪枝方法

**Decison Trees** 实现了正态分布数据的分类

## 朴素贝叶斯 Naive Bayes

应用NLP，中的N-gram， 前提是要时间相互独立，概率之间才可连乘，一般将连乘转为log，连加

**bayes_salary**: 构建朴素贝叶斯分类器，预测一个人的薪水是否达到50K/Y： 采用onehot编码

依据数据类型可分为三类：

* 高斯贝叶斯

遇到连续属性值的情况，假设每个分类和高斯或正态分布相关，

![贝叶斯](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQEWCcq1XtC1Yw20KWSHn2axYa7eY-a0T1TGtdVn5PvOpv9wW3FeA&s)

* 多项式贝叶斯分布

可应用于文本分类情况，样本特征向量代表频率时间，pi 代表事件发生i次
P(x_{i}|y_{k})=\frac{N_{y_{k},x_{i}}+\alpha}{N_{y_{k}}+n\alpha}

情况：当事件未在样本中出现时，即属性值为0的离散情况，平滑处理，赋值一个概率。当α=1α=1时，称作Laplace平滑，当0<α<10<α<1时，称作Lidstone平滑，α=0α=0时不做平滑。

* 伯努利模型

伯努利模型中，条件概率P(xi|yk)P(xi|yk)的计算方式是：

当事件发生过，即特征值xixi为1时，$$P(xi|yk)=P(xi=1|yk)P(xi|yk)=P(xi=1|yk)；$$

当事件未发生过，即特征值xixi为0时，$$P(xi|yk)=1−P(xi=1|yk)P(xi|yk)=1−P(xi=1|yk)；$$



## K近邻  K Neaerest Neighbour
## 回归  Regression 逻辑回归

## 支持向量机 Support Vector Machines



# 无监督学习
## 聚类 clustering problem
## 相似度量 Similarity measures
## K均值算法 The K-means algorithm

# 强化学习 

