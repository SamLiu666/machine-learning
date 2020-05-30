# 泰戈尔风格的文本生成模型

参考：tf 官方教程 https://www.tensorflow.org/tutorials/text/text_generation 莎士比亚文本风格生成其

# 1 数据和数据处理

爬取泰格诗集的数据，整理合并成一个语料库，进行数据处理，同时练习word2vec模型，相似词

corpus.txt:  所有诗词的文本字符串

segment.txt：jieba 分词之后保存的文件

## 1.2 word2vec 应用

word2vec 训练语料（segment.txt），得到词向量模型，保存模型，调用，得到一些最相似的词的小应用

# 2 模型训练

## 2.1 生成模型所需数据

再一次处理泰戈尔诗集文本内容，进行序列化。同样创建两个字典方便转换，具体可查看文档

## 2.2 tensorflow 2.0 使用LSTM 和 GRU 创建模型

第一次只用了GRU和10次迭代，效果不佳

第二次尝试 LSTM+GRU，神经元参数达到千万

![z](https://github.com/littlebeanbean7/docs/blob/master/site/en/tutorials/text/images/text_generation_sampling.png?raw=1)



创建训练存储点，训练模型 

# 3 测试模型

导入存储点，生成文本，输入起始句子，生成指定长度的文本

**测试结果：**相对第一次的结果，这次更加像文本的结构，有段落。句子的可读性更好！

优化方向：

- 文本数据集没有清理干净
- 泰戈尔的诗歌和小说混合，可能影响了结果
- 诗歌的词与词的分离，空行之间的平衡问题

# 4.模型部署

线上部署模型。

这个模型已经训练完成，每次只需调用检查点即可