# 动手学机器学习

学习机器学习原理，公式推导，实践入门级别的机器学习项目，实现机器学习算法。深度学习相关算法和应用

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

[补充](https://blog.csdn.net/weixin_36586536/article/details/80468426)

依据分类原则可分为两类：基尼系数，熵

| 算法 | 划分标准   |
| ---- | ---------- |
| ID3  | 信息增益   |
| C4.5 | 信息增益率 |
| CART | 基尼系数   |

![1](http://www.learnbymarketing.com/wp-content/uploads/2016/02/entropy-formula.png)
![2](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRzYHkcmZKKp2sJN1HpHvw-NgqbD9EnapnbXozXRgajrSGvEnYy&s)

### 三要素

#### 特征选择

有三种方法进行特征选择：ID3: 信息增益，C4.5: 信息增益比，CART: 基尼系数

 $$ 信息熵：H(D) = - \sum_{k=1}^k \frac{|C_k|}{|D|} log_2 \frac{|C_k|}{|D|} \\ 条件熵： H(D|A) = \sum_{i=1}^n \frac{|D_i|}{|D|} H(D_i) \\ 信息增益： g(D,A)=H(D)-H(D|A) $$

**信息增益比本质：**在信息增益的基础之上乘上一个惩罚参数。特征个数较多时，惩罚参数较小；特征个数较少时，惩罚参数较大。

 $$ 信息增益比 = 惩罚参数 \times 信息增益 \ 信息增益比：g_R(D,A) = \frac{g(D,A)}{H(D)} $$

CART 既可以用于分类，也可以用于回归:

 $$ Gini(D) = \sum_{k=1}^{|K|}p_k(1-p_k)= 1 - \sum_{k=1}^{K} p_k^2 $$ 

$$ Gini(D, A=a) = \frac{D_1}{D}Gini(D_1) + \frac{D_2}{D} Gini(D_2) $$

#### 剪枝处理

决策树学习算法用来**解决过拟合**的一种办法

- 预剪枝：决策树生成过程中，在每个节点划分前先估计其划分后的泛化性能， 如果不能提升，则停止划分，将当前节点标记为叶结点。
- 后剪枝：生成决策树以后，再自下而上对非叶结点进行考察， 若将此节点标记为叶结点可以带来泛化性能提升，则修改之。

**Decison Trees** 实现了正态分布数据的分类

### 随机森林

可应用于回归问题，也可应用于分类问题。随机森林，可理解为组合一些决策树，**根据多个训练集与特征集合来建立多颗决策树，然后进行投票决策**

随机森林的最终目的是建立 m 颗决策树，而每颗决策树的建立过程如下：

- 如果训练集大小为N，对于每棵树而言，**随机**且有放回地从训练集中的抽取N个训练样本，作为该树的训练集。
- 如果每个样本的特征维度为M，指定一个常数m<<M，**随机**地从M个特征中选取m个特征子集，每次树进行分裂时，从这m个特征中选择最优的
- 每棵树都尽最大程度的生长，并且没有剪枝过程。

随机森林中的随性性指的是：**数据采样的随机性与特征采用的随机性。** 这两个随机性的引入对随机森林的分类性能直观重要，它们使得随机森林不容易陷入过拟合，且具有很好的抗噪能力

![随机森冷](https://i.ytimg.com/vi/goPiwckWE9M/maxresdefault.jpg)

通过交叉验证来调整树的数量，解决过拟合问题

## 朴素贝叶斯 Naive Bayes

应用NLP，中的N-gram， 前提是要时间相互独立，概率之间才可连乘，一般将连乘转为log，连加

条件概率：  $$ P(X|Y) = \frac{P(X,Y)}{P(Y)} $$

- 先验概率：表示事件发生前的预判概率，一般都是单独事件发生的概率，如 P(A)
- 后验概率：依据数据类型可分为三类：基于先验概率求得的**反向条件概率**，形式上与条件概率相同（若 `P(X|Y)` 为正向，则 `P(Y|X)` 为反向）

贝叶斯公式： $$ P(Y|X) = \frac{P(X|Y) P(Y)}{P(X)} \ $$

- P(Y) 叫做**先验概率**，意思是事件X发生之前，我们对事件Y发生的一个概率的判断
- P(Y|X) 叫做**后验概率**，意思是时间X发生之后，我们对事件Y发生的一个概率的重新评估
- P(Y,X) 叫做**联合概率**， 意思是事件X与事件Y同时发生的概率。

条件独立假设： $$ P(x|c) = p(x_1, x_2, \cdots x_n | c) = \prod_{i=1}^Np(x_i | c) $$  -》简化运算



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

K 离谁近就取谁的值

## 回归  Regression 逻辑回归

解决分类问题，一般是离散分类问题

**Sigmoid 函数**  
![sigmoid](https://miro.medium.com/max/970/1*Xu7B5y9gp0iL5ooBj7LtWw.png)

决策边界，>=0.5 class==1, <0.5 class=0 

前提条件，二项式多项式。例子：预测明天是否会下雨

### 逻辑回归

- 本质极大似然估计：maximum likelihood estimation，缩写为MLE

- 逻辑回归的激活函数：Sigmoid

  $$ h_\theta(x) = sigmoid(\theta^T X) = \frac{1}{1 + e^{-\theta^T X}} $$

- 逻辑回归的代价函数：交叉熵

  $$ Cost(h_{\theta}(x),y) = \begin{cases} -log(h_{\theta(x)}) & if , y = 1\ -log(1-h_{\theta(x)}) & if , y = 0 \end{cases} $$

求解：

$$ J(\theta) = - \frac{1}{m} \left[ \sum_{i=1}^m y^{(i)}log(h_\theta(x^{(i)})) + (1-y^{(i)}) log(1 - h_\theta(x^{(i)})) \right] $$

 $$ J(\theta) = - \frac{1}{m} \left[ \sum_{i=1}^m y^{(i)}log(h_\theta(x^{(i)})) + (1-y^{(i)}) log(1 - h_\theta(x^{(i)})) \right] + \frac{\lambda}{2m} \sum_{j=1}^{m}\theta_j^2 $$  （L2 正则化）

使用梯度下降算法去求解最小值时对应的参数

### 线性回归

 $$ 线性回归：f(x)=\theta ^{T}x \ 逻辑回归：f(x)=P(y=1|x;\theta )=g(\theta ^{T}x)， \quad g(z)=\frac{1}{1+e^{-z}} $$

线性回归其参数计算方式为**最小二乘法**， 逻辑回归其参数更新方式为**极大似然估计**。

- sigmoid : $$ g(z) = \frac{1}{1+e^{-z}} \ g'(z) = g(z)(1-g(z)) $$
- LR 的定义： $$ h_{\theta}(x) = g(\theta^Tx) = \frac{1}{1 + e^{-\theta^Tx}} $$
- 损失函数： $$ J(\theta) = - \frac{1}{m} \left[ \sum_{i=1}^m y^{(i)}log(h_\theta(x^{(i)})) + (1-y^{(i)}) log(1 - h_\theta(x^{(i)})) \right] $$

- 岭回归本质上是 **线性回归 + L2 正则化**。 $$ \hat{h}*{\theta}(x) = h*{\theta}(x) + \lambda \sum_i w_i^2 $$

- Lasso 回归的本质是 **线性回归 + L1 正则化**。 $$ \hat{h}*{\theta}(x) = h*{\theta}(x) + \lambda \sum_i |w_i| $$
- ElasticNet 回归 本质上是线性回归 + L1正则化 + L2 正则化。

最小二乘法估计：

$$ X = (x_1, ..., x_N)^T \ Y = (y_1, ..., y_N)^T $$

$$ \begin{align} L(w) &= \sum_{i=1}^N ||w^Tx_i - y_i ||^2 \ &= \sum_{i=1}^N (w^Tx_i - y_i)^2 \ &= \begin{pmatrix} w^Tx_1 - y_1 & ... & w^Tx_N - y_N \end{pmatrix} \begin{pmatrix} w^Tx_1 - y_1 \ ... \ w^Tx_N - y_N \end{pmatrix} \end{align} $$

其中有： $$ \begin{align} \begin{pmatrix} w^Tx_1 - y_1 & ... & w^Tx_N - y_N \end{pmatrix} = w^T \begin{pmatrix} x_1 & ... & x_N \end{pmatrix} - \begin{pmatrix} y_1 & ... & y_N \end{pmatrix} &= w^TX^T - Y^T \end{align} $$ $$\begin{align} \begin{pmatrix} w^Tx_1 - y_1 \ ... \ w^Tx_N - y_N \end{pmatrix} = \begin{pmatrix} x_1 \ ... \ x_N \end{pmatrix}w - \begin{pmatrix} y_1 \ ... \ y_N \end{pmatrix} = Xw-Y \end{align}$$

那 么，最终就得到： $$ \begin{align} L(w) = (w^TX^T - Y^T)(Xw + Y) \ &= w^TX^TXw - w^TX^TY - Y^TXw - Y^TY \end{align} $$ 考虑到 $w^TX^TY$ 与 $Y^TXw$ 的结果其实都是一维实数且二者为转置，因此，二者的值相等， 那么就有： $$ L(w) = w^TX^TXw - 2w^TX^TY - Y^TY $$ 那么就有： $$ \hat{w} = argmin , L(w) \ \frac{\delta L(w)}{\delta w} = 2X^TXw - 2X^TY = 0 $$ 从而就得到： $$ w = (X^TX)^{-1}X^TY $$



## 支持向量机 Support Vector Machines

[Kaggle SVM torturial](https://www.kaggle.com/prashant111/svm-classifier-tutorial)

SVM 三宝： **间隔，对偶，核技巧**。它属于**判别模型**，可应用于分类和回归问题。SVM可用于线性分类问题，也可以通过核方法用于非线性分类问题

![svm](https://static.wixstatic.com/media/8f929f_7ecacdcf69d2450087cb4a898ef90837~mv2.png)

最大边界超平面

![margin](https://static.packt-cdn.com/products/9781783555130/graphics/3547_03_07.jpg)

核方法，超平面映射

![kernel](http://www.aionlinecourse.com/uploads/tutorials/2019/07/11_21_kernel_svm_3.png)

核 将低纬度的数据，转换到高维度的空间 ![和函数](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTodZptqcRor0LGo8Qn7_kJB9n9BACMt6jgIPZ4C3g_rgh_uSRZLQ&s)

分类：**linear kernel : K(xi , xj ) = xiT xj** ,  **Polynomial kernel : K(xi , xj ) = (γxiT xj + r)d , γ > 0** , **Radial Basis Function Kerne** ,   **sigmoid kernel : k (x, y) = tanh(αxTy + c)**

- 线性可分

$D_0$ 和 $D_1$ 是 n 维空间中的两个点集， 如果存在 n 维向量 $w$ 和实数 $b$ ， 使得：$$ wx_i +b > 0; \quad x_i \in D_0 \ wx_j + b < 0; \quad x_j \in D_1 $$ 则称 $D_0$ 与 $D_1$ 线性可分。

- 最大间隔超平面

能够将 $D_0$ 与 $D_1$ 完全正确分开的 $wx+b = 0$ 就成了一个超平面。

为了使得这个超平面更具鲁棒性，我们会去找最佳超平面，以最大间隔把两类样本分开的超平面，也称之为**最大间隔超平面**。

- 两类样本分别分割在该超平面的两侧
- 两侧距离超平面最近的样本点到超平面的距离被最大化了

# 无监督学习

**定义** 从无标签的数据中获取信息  $$ f: x  {->} y$$ ，发现输入之间的练习，不依靠任何学习过程中的任何反馈

分类

- 无监督特征学习： 从无标签的训练数据中挖掘有效的特征和表示，一般用来降维、数据可视化或监督学习的前期处理
- 概率密度估计
  - 参数密度估计： 先假设服从某个概率分布，然后用样本训练去估计参数
  - 非参数估计： 利用训练样本对密度进行估计
- 聚类： 将一组样本按照一定准则划分到不同组（簇），通用准则-组内样本相似性高于组间样本的相似性，常见包括：K-Means等

## 相似度计算

基于勾股定理（集合距离）

## 主成分分析法 

[PCA -Principal Component Analysis ](https://www.matongxue.com/madocs/1025/)

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

### 相似度量 Similarity measures

$$ s(x_i, x_j) = - ||x_i - x_j||^{2} $$  相似特性用负数的距离平方表示

$$ r_{i,k} \leftarrow s_(x_i, x_k) - \max_{k' \neq k} \left\{ a_{i,k'} + s(x_i, x_k') \right\} $$  

### 准确性指标

n-样本观测值, a-相同簇类的观测值, b-不同簇类观测值

$$\Large \text{RI} = \frac{2(a + b)}{n(n-1)}$$

ARI指标

$$\Large \text{ARI} = \frac{\text{RI} - E[\text{RI}]}{\max(\text{RI}) - E[\text{RI}]}$$ 

## 簇类 Clustering

### K均值算法 The K-means algorithm

$$\Large J(C) = \sum_{k=1}^K\sum_{i~\in~C_k} ||x_i - \mu_k|| \rightarrow \min\limits_C$$

随机给定初始中心点，计算数据中每一个点到中心点的距离，重复直到簇类中心点和数据的均值据购销

C-类集合中的K簇的具体类，

$$\Large D(k) = \frac{|J(C_k) - J(C_{k+1})|}{|J(C_{k-1}) - J(C_k)|}  \rightarrow \min\limits_k$$

## 神经网络 Neural Network

Winner-Take-All Networks：输出值概率高对应的权重值增加

Counter-Propagation Networks

Kohonen’s Self-Organising Feature Maps SOMs



# 半监督学习

semi-supervised learning: 监督学习和无监督学习的混合

# 深度学习

定义：直接从数据中获取信息，基于人工神经网络（ANN），属于机器学习的一部分 [扩展](https://www.rsipvision.com/exploring-deep-learning/)

![dl](https://www.rsipvision.com/wp-content/uploads/2015/04/Slide5.png)

![dlll](https://www.rsipvision.com/wp-content/uploads/2015/04/Slide4.png)

需要大量数据，训练时间较长，小样本数据性能不如机器学习

逻辑回归，计算图，参数初始化， Forward Propagation（正向传播）

Optimization Algorithm with Gradient Descent - 梯度下降优化算法

## 神经网络

ANN Artificial Neural Network：  使机器像人脑一样，能够从模式中学习，能够基于学习内容推导新情况的状态

**交叉熵损失函数**

$$J = - \frac{1}{m} \sum\limits_{i = 0}^{m} \large\left(\small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right)  \large  \right) \small$$

### 优点

-  Generalisability 普遍性：对于未出现过的数据预测更准确
-  Fault Tolerance 容错性：对噪声容忍度更大
- Determine what is to be classified or predicted 

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

**LeNet 模型**

![ccc](https://tangshusen.me/Dive-into-DL-PyTorch/img/chapter05/5.5_lenet.png)

**AlexNet**

![model](https://tangshusen.me/Dive-into-DL-PyTorch/img/chapter05/5.6_alexnet.png)



## 循环神经网络

RNN--Recurrent Neural Network： 本质上是重复ANN，从先前的非线性激活函数传递信息， 语言模型

$$ p(w_1, w_2, w_3, w_4) = p(w_1)p(w_2|w_1)p(w_3|w_1,w_2)p(w_4|w_1,w_2,w_3)$$

n元语法- n-gram：马尔可夫假设是指一个词的出现只与前面nn*n*个词相关，即nn*n*阶马尔可夫链（Markov chain of order nn*n*）。

一元模型- $$ p(w_1, w_2, w_3, w_4) = p(w_1)p(w_2)p(w_3)p(w_4)$$

二元模型：$$ p(w_1, w_2, w_3, w_4) = p(w_1)p(w_2|w_1)p(w_3|w_2)p(w_4|w_3)$$

三元模型：$$ p(w_1, w_2, w_3, w_4) = p(w_1)p(w_2|w_1)p(w_3|w_1,w_2)p(w_4|w_1,w_2,w_3)$$

![yicang](https://trickygo.github.io/Dive-into-DL-TensorFlow2.0/img/chapter06/6.2_rnn.svg)

隐藏状态的RNN

### 基于字符级RNN的语言模型

![RNN](https://trickygo.github.io/Dive-into-DL-TensorFlow2.0/img/chapter06/6.2_rnn-train.svg)

RNN ：对应的类型- N-N, N-1, 1-N, N-M

其中N-M 又称为 Encoder-Decoder 模型，也叫Seq2Seq

仔细分析普通的RNN模型，我们发现它有一个缺陷：它只能记住前面的输入对它的影响，而不能将后面的输入对它的影响记忆下来。

于是产生了双向RNN。

### RNN 梯度消失问题

RNN一个最大的缺陷就是**梯度消失与梯度爆炸问题，** 由于这一缺陷，使得RNN在长文本中难以训练， 这才诞生了LSTM及各种变体。

梯度消失和梯度爆炸产生问题：

1. 步伐太大，权重更新太快，可适当减少学习率的大小
2. 学习不稳地，无法获取最优参数，无法从数据中学习，甚至权重变为NaN而无法更新
3. 使i网络不稳定

原因：sigmoid 反向传播求导的函数值随层数增加，越来越小

无论是梯度消失还是梯度爆炸，都是**源于网络结构太深**，造成网络权重不稳定，从本质上来讲是**因为梯度反向传播中的连乘效应**

**LSTM** 解决梯度爆炸和梯度消失问题：

- LSTM通过门机制完美的解决了这两个问题

## 长短期记忆

Long-Short Term Memory :LSTM 中引入了3个门，即输入门（input gate）、遗忘门（forget gate）和输出门（output gate），以及与隐藏状态形状相同的记忆细胞（某些文献把记忆细胞当成一种特殊的隐藏状态），从而记录额外的信息

如果门的输出是0， 就表示将门紧紧关闭，为1则表示将门完全打开，而位于0-1之间的实数表示将门半开，至于开的幅度跟这个数的大小有关，门就表示变量对变量的的影响程度。

![ss](https://trickygo.github.io/Dive-into-DL-TensorFlow2.0/img/chapter06/6.8_lstm_0.svg)

遗忘门

![a](https://www.zhihu.com/equation?tex=f_t+%3D+%5Csigma%7B%28W_f+%5Ccdot+%5Bh_%7Bt-1%7D%2Cx_t%5D+%2B+b_f%29%7D+)

![[公式]](https://www.zhihu.com/equation?tex=f_t) ： 遗忘门输出，用于控制上一时刻的单元状态 ![[公式]](https://www.zhihu.com/equation?tex=c_%7Bt-1%7D) 有多少保留到当前时刻 ![[公式]](https://www.zhihu.com/equation?tex=c_t) 。

**输入门决定了当前时刻网络的输入** ![[公式]](https://www.zhihu.com/equation?tex=x_t) **有多少保存到单元状态** ![[公式]](https://www.zhihu.com/equation?tex=c_t) **。**

![[公式]](https://www.zhihu.com/equation?tex=+i_t%3D%5Csigma%28W_i%5Ccdot%5Bh_%7Bt-1%7D%2Cx_t%5D%2Bb_i%29+)

![[公式]](https://www.zhihu.com/equation?tex=i_t) ： 输入门的输出值，是一个0 - 1 之间的实数，决定了当前时刻网络的输入 ![[公式]](https://www.zhihu.com/equation?tex=x_t) 有多少保存到单元状态 ![[公式]](https://www.zhihu.com/equation?tex=c_t) 。

输出门就是**用来控制单元状态** ![[公式]](https://www.zhihu.com/equation?tex=+c_t) **有多少输入到 LSTM 的当前输出值** ![[公式]](https://www.zhihu.com/equation?tex=h_t) **。**

![[公式]](https://www.zhihu.com/equation?tex=+o_t%3D%5Csigma%28W_o%5Ccdot%5Bh_%7Bt-1%7D%2Cx_t%5D%2Bb_o%29+)

LSTM的重点在于思想， 有很多paper都有介绍LSTM的变体: **普通RNN的隐层只有一个状态，如**h**，该状态对短期的输入十分敏感，这使得RNN处理短期依赖问题很拿手，那么如果我们再添加一个状态，如**c**，让它来保存长期的状态，这样，我们不就能保证对长期的输入保持一定的敏感，问题不就解决了。**



**Steps of LSTM:**

LSTM 的循环神经网络

![lstm](https://github.com/TrickyGo/Dive-into-DL-TensorFlow2.0/raw/master/docs/img/chapter06/6.5.png)

具体到NLP领域中，通常我们会在 Embedding 层采用双向 LSTM，GRU 对文本进行扫描来获得每个词的上下文表示， 如果我们采用传统的RNN，这就意味着句子的第一个单词对句子最后一个单词不起作用了，这明显是有问题的。

## GRU

GRU 抛弃了 LSTM 中的 ![[公式]](https://www.zhihu.com/equation?tex=h_t) ，它认为既然 ![[公式]](https://www.zhihu.com/equation?tex=c_t) 中已经包含了 ![[公式]](https://www.zhihu.com/equation?tex=h_t) 中的信息了，那还要 ![[公式]](https://www.zhihu.com/equation?tex=h_t) 做什么，于是，它就把 ![[公式]](https://www.zhihu.com/equation?tex=h_t) 干掉了。 然后，GRU 又发现，在生成当前时刻的全局信息时，我当前的单元信息与之前的全局信息是此消彼长的关系，直接用 ![[公式]](https://www.zhihu.com/equation?tex=1-z_t) 替换 ![[公式]](https://www.zhihu.com/equation?tex=i_t) ，简单粗暴又高效 。

![img](https://pic3.zhimg.com/80/v2-805c262b34d643449d42efdd8f3be49a_720w.jpg)

归纳一下 LSTM 与 GRU 的区别：

- 首先， LSTM 选择暴露部分信息（ ![[公式]](https://www.zhihu.com/equation?tex=h_t) 才是真正的输出， ![[公式]](https://www.zhihu.com/equation?tex=c_t) 只是作为信息载体，并不输出)， 而GRU 选择暴露全部信息。
- 另一个区别在于输出变化所带来的结构调整。为了与LSTM的信息流保持一致，重置门本质上是输出门的一种变化，由于输出变了，因此其调整到了计算 ![[公式]](https://www.zhihu.com/equation?tex=h%27_t) 的过程中。

对于 LSTM 与 GRU 而言， 由于 GRU 参数更少，收敛速度更快，因此其实际花费时间要少很多，这可以大大加速了我们的迭代过程。 而从表现上讲，二者之间孰优孰劣并没有定论，这要依据具体的任务和数据集而定，而实际上，二者之间的 performance 差距往往并不大，远没有调参所带来的效果明显，与其争论 LSTM 与 GRU 孰优孰劣， 不如在 LSTM 或 GRU的激活函数（如将tanh改为tanh变体）和权重初始化上功夫。

# 强化学习

reinforcement learning:  通过环境判断获取信息是否正确，不断调整获取答案的模型

# 相关概念

## 激活函数

### softmax



### Relu

$$ReLU(x)=max(x,0)$$

最常用的深度学习激活函数，只能在隐藏层使用

### Sigmoid

$$sigmoid(x)=\frac{1}{1+exp(−x)}$$

通常应用在分类问题，避免梯度消失问题

### tanh

$$tanh(x)=\frac{1+exp(−2x)}{1+exp(−2x)}$$

避免梯度消失问题

## 模型复杂度

$$y=b+\sum_{k=1}^{K} {x^k w_k }$$

多项式函数拟合的目标是找一个KK*K*阶多项式函数,*w**k*是模型的权重参数，bb*b*是偏差参数。与线性回归相同，多项式函数拟合也使用平方损失函数。

## 正则化

Regularization： 欠拟合需要更好的训练数据，过拟合需要更好的泛化能力， 目的是为了优化这两种情况

### 拉素回归  L1-  lasso

Lasso regression: 也叫L1 正则化， 使用绝对值，数据中含有许多不相关数据

$$Lasso regression lost fuction = OLS + alpha *$$

### 岭回归 L2 - weight decay

Ridge regression: 也叫L2 正则化，使用平方值，惩罚较大的权重值，对于大部分分类和预测问题

$$ Ridge regression lost fuction = OLS + alpha * sum(parameter^2)$$

## Dropout

减少了过拟合，降低了训练误差和测试误差。 原理：在训练中随机丢失一些神经元，每一个神经元相互独立固定概率 p

超参数一般选择p=0.2  或 p=0.5，神经元丢失概率

## 调参

HYPERPARAMETER TUNING 

## 欠拟合过拟合

欠拟合：在训练数据上表现不好，

过拟合：在训练数据上表现很好，测试数据表现不佳，泛化能力不行

## Confusion Matrix

• Recall (also called sensitivity or the true positive rate) 召回率
= TP / (TP+FN) ( = 636 / (636+64) = 90.9%)
– A measure of a classifier’s completeness.
– A classifier that produces no false negatives has a recall of 1.0.
• Precision (also called the positive predictive value) 精度
= TP / (TP+FP) ( = 636 / (636+160) = 79.9%)
– A measure of a classifier’s exactness.
– A classifier that produces no false positives has a precision of 1.0.

• TP: the classifier predicts it is a YES and it truly is a YES.
• FN: the classifier predicts it is a NO but it actually is a YES.
• FP: the classifier predicts it is a YES but it actually is a NO.
• TN: the classifier predicts it is a NO and it truly is a NO.

## ROC curve or AUC-ROC curve

**接收者操作特征曲线**（**receiver operating characteristic curve**，或者叫**ROC曲线**）是一种坐标图式的分析工具，用于 (1) 选择最佳的信号侦测模型、舍弃次佳的模型。 (2) 在同一模型中设定最佳阈值。

.演示二进制文件性能的图形图表具有不同决策(识别)阈值的分类器设置。

# 应用

## NLP

NLP ， DL class 查看对应的名称文件

## CV

DM-class 相关课程

## RS

并未涉及

书籍阅读

keep working