CS244N 课程-2019 学习记录

课程主页：http://web.stanford.edu/class/cs224n/

lecture：https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z

课后作业参考代码：https://blog.csdn.net/zyx_ly/article/details/100594378

CS231N ： 

网站主页 http://cs231n.stanford.edu/syllabus.html





## 1 word vector

1. [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
2. [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf) (original word2vec paper)
3. [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) (negative sampling paper)

**Assignment 1 **：Exploring Word Vectors (word embedding)

1. co-occurrence matrices：共现矩阵记录了两个词同时出现的次数，词和词的矩阵$M_{i,j}$，$w_j$ 出现在$w_i$的次数， SVD 奇异值分解降维，实现过程

   - 提取语料库中的所有词(word)
   - Numpy 实现共现矩阵（窗口大小为4）
   - SVD 降维

2.  word2vec： sg=0 or 1 选择skpi-gram or cbow 

   - 训练自己的语料库，神雕侠侣和泰戈尔诗集（爬取数据，并分词处理）

   - 调用gensim.models  的 word2vec ，训练模型，保存并加载模型，

     ```python
     model.save('天龙八部.model')
     model = word2vec.Word2Vec.load('天龙八部.model')
     ```

   - 测试，使用，查看效果

     ```python
     wv_from_bin.most_similar("column")
     w1_w2_dist = wv_from_bin.distance(w1, w2)
     pprint.pprint(wv_from_bin.most_similar(positive=['she', 'him'], negative=['he']))
     ```

## 2 Word Vectors 2 and Word Senses

Suggested Readings:

1. [GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/pubs/glove.pdf) (original GloVe paper)
2. [Improving Distributional Similarity with Lessons Learned from Word Embeddings](http://www.aclweb.org/anthology/Q15-1016)
3. [Evaluation methods for unsupervised word embeddings](http://www.aclweb.org/anthology/D15-1036)

Additional Readings:

1. [A Latent Variable Model Approach to PMI-based Word Embeddings](http://aclweb.org/anthology/Q16-1028)
2. [Linear Algebraic Structure of Word Senses, with Applications to Polysemy](https://transacl.org/ojs/index.php/tacl/article/viewFile/1346/320)
3. [On the Dimensionality of Word Embedding.](https://papers.nips.cc/paper/7368-on-the-dimensionality-of-word-embedding.pdf)

**Assignment 2**：word2vec 理论和实现

理论部分： 

1. word2vec 两个模型中心思想：通过词预测词（连续词-》单个；单个-》上下文）
2. 最大似然估计，损失函数，交叉熵损失函数，梯度计算推导

实践部分：

1. word2vec模型实现(numpy)，损失函数，梯度，反向传播求梯度
2. 随机梯度下降算法实现
3. 语料训练接口，改进尝试英文分词、中文