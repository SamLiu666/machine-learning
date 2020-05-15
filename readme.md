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

**定义** 从无标签的数据中获取信息  $$ f: x  {->} y$$

分类

- 无监督特征学习： 从无标签的训练数据中挖掘有效的特征和表示，一般用来降维、数据可视化或监督学习的前期处理
- 概率密度估计
  - 参数密度估计： 先假设服从某个概率分布，然后用样本训练去估计参数
  - 非参数估计： 利用训练样本对密度进行估计
- 聚类： 将一组样本按照一定准则划分到不同组（簇），通用准则-组内样本相似性高于组间样本的相似性，常见包括：K-Means等

## 主成分分析法 PCA -Principal Component Analysis

定义：数据降维方法，使得在转换后的空间数据的方差最大

### （线性）编码

给定一组基向量 $$A = [a_1, a_2, ..., a_M]$$ 将输入样本表示为基向量组成的线性组合 $$ x=Az$$ (A-dictionary, z-encoding)

* 完备性：M个向量支撑M维的欧式空间
* 冗余：M > D， M个基向量支撑 D维空间

冗余-过完备： 基向量不具备独立正交的性质，需要进行处理

- 稀疏编码（Sparse Coding）
- 自编码器（Auto-Encoder)
- 稀疏自编码器
- 降噪自编码器

## 概率密度估计

### 参数密度估计 - Parametric Density Estimation

根据先验知识，假设随机变量服从某种分布，然后通过训练样本估计分布的参数，估计方法：最大似然估计MLE

$$log {p(D; \Theta) }= \sum_{n=1}^{N}log {p{(x^{(n)}; \Theta)}} $$

正态分布，多项分布 等等

**存在问题**：

* 模型选择
* 不可观测变量
* 维度灾难

### 非参数密度估计（Nonparametric Density Estimation

$$ p(x) = {\frac{K}{NV}}$$  

N - N个训练样本， K - 落入R区域的样本数量服从二项分布， 条件 N足够大， R区域足够小

**方法**

* 直方图法-柱状图
* 核密度估计（Kernel Density Estimation）
* K 近邻方法

## 聚类 Clustering

### K均值算法 The K-means algorithm

$$\Large J(C) = \sum_{k=1}^K\sum_{i~\in~C_k} ||x_i - \mu_k|| \rightarrow \min\limits_C$$

C-类集合中的K簇的具体类，

$$\Large D(k) = \frac{|J(C_k) - J(C_{k+1})|}{|J(C_{k-1}) - J(C_k)|}  \rightarrow \min\limits_k$$



### 相似度量 Similarity measures

$$ s(x_i, x_j) = - ||x_i - x_j||^{2} $$  相似特性用负数的距离平方表示

$$ r_{i,k} \leftarrow s_(x_i, x_k) - \max_{k' \neq k} \left\{ a_{i,k'} + s(x_i, x_k') \right\} $$  

### 准确性指标

n-样本观测值, a-相同簇类的观测值, b-不同簇类观测值

$$\Large \text{RI} = \frac{2(a + b)}{n(n-1)}$$

ARI指标

$$\Large \text{ARI} = \frac{\text{RI} - E[\text{RI}]}{\max(\text{RI}) - E[\text{RI}]}$$ 



# 深度学习

定义：直接从数据中获取信息

逻辑回归，计算图，参数初始化， Forward Propagation（正向传播）

Optimization Algorithm with Gradient Descent - 梯度下降优化算法

## 逻辑回归

## 神经网络

ANN Artificial Neural Network 

**交叉熵损失函数**

$$J = - \frac{1}{m} \sum\limits_{i = 0}^{m} \large\left(\small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right)  \large  \right) \small$$



## 卷积神经网络

convolutional neural network--CNN

**相关运算**

![cnn](https://tangshusen.me/Dive-into-DL-PyTorch/img/chapter05/5.1_correlation.svg)

$$0×0+1×1+3×2+4×3=19\\1×0+2×1+4×2+5×3=25\\3×0+4×1+6×2+7×3=37\\4×0+5×1+7×2+8×3=43$$

**多输入通道**

![c](https://tangshusen.me/Dive-into-DL-PyTorch/img/chapter05/5.3_conv_multi_in.svg)

**池化池**

![cc](https://tangshusen.me/Dive-into-DL-PyTorch/img/chapter05/5.4_pooling.svg)

取最大值：$$max(0,1,3,4)=4,max(1,2,4,5)=5,max(3,4,6,7)=7,max(4,5,7,8)=8.$$

### 卷积神经网络

![ccc](https://tangshusen.me/Dive-into-DL-PyTorch/img/chapter05/5.5_lenet.png)

### 

* LeNet 模型

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

## 正则化

### 岭回归 

Ridge regression: 也叫L2 正则化，

$$ Ridge regression lost fuction = OLS + alpha * sum(parameter^2)$$

### 拉素回归

Lasso regression: 也叫L1 正则化

$$Lasso regression lost fuction = OLS + alpha *$$

## ROC

**接收者操作特征曲线**（**receiver operating characteristic curve**，或者叫**ROC曲线**）是一种坐标图式的分析工具，用于 (1) 选择最佳的信号侦测模型、舍弃次佳的模型。 (2) 在同一模型中设定最佳阈值。

## 调参

HYPERPARAMETER TUNING 