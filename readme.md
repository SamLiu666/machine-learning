# 动手学机器学习

此项目用于学习机器学习原理，公式推导，实践入门级别的机器学习项目，实现机器学习算法。

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

### 随机森林

可应用于回归问题，也可应用于分类问题。随机森林，可理解为组合一些决策树

![随机森冷](https://i.ytimg.com/vi/goPiwckWE9M/maxresdefault.jpg)



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

KNN 是一种应用于分类问题和逻辑回归的非参数算法，也是一种懒惰算法（不需要为建立模型训练数据）。一般使用欧几里得或者曼哈顿举例

*使用交叉验证取K值* 

## 回归  Regression 逻辑回归

解决分类问题，一般是离散分类问题

**Sigmoid 函数**  
![sigmoid](https://miro.medium.com/max/970/1*Xu7B5y9gp0iL5ooBj7LtWw.png)

决策边界，>=0.5 class==1, <0.5 class=0 

前提条件，二项式多项式。例子：预测明天是否会下雨

## 支持向量机 Support Vector Machines

[Kaggle SVM torturial](https://www.kaggle.com/prashant111/svm-classifier-tutorial)

可应用于分类和回归问题。SVM可用于线性分类问题，也可以通过核方法用于非线性分类问题

![svm](https://static.wixstatic.com/media/8f929f_7ecacdcf69d2450087cb4a898ef90837~mv2.png)

最大边界超平面

![margin](https://static.packt-cdn.com/products/9781783555130/graphics/3547_03_07.jpg)

核方法，超平面映射

![kernel](http://www.aionlinecourse.com/uploads/tutorials/2019/07/11_21_kernel_svm_3.png)

核 将低纬度的数据，转换到高维度的空间 ![和函数](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTodZptqcRor0LGo8Qn7_kJB9n9BACMt6jgIPZ4C3g_rgh_uSRZLQ&s)

分类：**linear kernel : K(xi , xj ) = xiT xj** ,  **Polynomial kernel : K(xi , xj ) = (γxiT xj + r)d , γ > 0** , **Radial Basis Function Kerne** ,   **sigmoid kernel : k (x, y) = tanh(αxTy + c)**

# 无监督学习
## 聚类 clustering problem
## 相似度量 Similarity measures
## K均值算法 The K-means algorithm

# 深度学习

# 相关概念

## 激活函数

### softmax

### Relu

$$ReLU(x)=max(x,0)$$

### Sigmoid

$$sigmoid(x)=\frac{1}{1+exp(−x)}$$

### tanh

$$tanh(x)=\frac{1+exp(−2x)}{1+exp(−2x)}$$

## 模型复杂度

$$y=b+\sum_{k=1}^{K} {x^k w_k }$$

多项式函数拟合的目标是找一个KK*K*阶多项式函数,*w**k*是模型的权重参数，bb*b*是偏差参数。与线性回归相同，多项式函数拟合也使用平方损失函数。

