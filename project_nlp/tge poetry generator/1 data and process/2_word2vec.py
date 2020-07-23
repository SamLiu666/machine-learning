import os, copy
import re,jieba
from gensim.models import word2vec
import pprint


def get_string():
    path = r"E:\chrome download\paper\corpus\tge\t.txt"
    text = " "
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line == '\n':
                line = line.strip("\n")
            text += line
        # content = f.read()
        f.close()
    # content = content.split()
    # print(content,len(content))
    # res = copy.deepcopy(content)

    # 正则表达式处理文本
    r1 = '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'

    # for s in content:
    res = re.sub(r1, '', text)
    # print(res, '\n', type(res), len(res))
    print('file done')
    return res

def svae_corpus():
    text = get_string()
    # 存储语料
    with open('cor.txt','a', encoding="utf-8") as f:
        f.write(text)
        f.close()
    # print(text)
    print('Done')


def seg_word():
    svae_corpus()
    # jieba 分词
    f = open('cor.txt', 'r', encoding='utf-8')
    text = f.read()
    # text = " "
    # for line in f.readlines():
    #     if line == '\n':
    #         line = line.strip("\n")
    #     text += line

    segment = jieba.cut(text, cut_all=True)
    with open(r"segment.txt", 'a', encoding='utf-8') as ff:
        ff.write(" ".join(segment))
        ff.close()
    print("全模式： "+ "/".join(segment))


def build_word2vec():
    seg_word()
    # 加载语料，训练模型
    sentences = word2vec.Text8Corpus('segment.txt')

    # 训练模型
    model = word2vec.Word2Vec(sentences)

    # 选出最相似的10个词
    for e in model.most_similar(positive=['张'], topn=10):
       print(e[0], e[1])

    # 保存模型
    model.save('tge.model')

    # load model
    model = word2vec.Word2Vec.load('tge.model')

    # 查找相似度
    print(model.similarity('你', '我'))

if __name__ == '__main__':
    model = word2vec.Word2Vec.load('tge.model')
    # for e in model.most_similar(positive=['我'], topn=10):
    #    print(e[:], e[:])
    print(model.most_similar("夏天"))
    print(model.most_similar("爱情"))
    pprint.pprint(model.most_similar(positive=['西方', '太阳'], negative=['上帝']))