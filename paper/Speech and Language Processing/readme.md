https://web.stanford.edu/~jurafsky/slp3/  inference

## 2.1 Regular Expressions

### traditional text processing

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