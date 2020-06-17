# path = r"D:\machine learning\project\0_chinese_poiet\1_corpus\segment_results\jieba_set.txt"
# # f = open(path, "r", encoding="utf-8")
# with open(path, "r", encoding="utf-8") as f:
#     contents = []
#     for line in f:
#         contents.append(line)
#     f.close()
# print(contents)

from gensim.models import word2vec

from gensim.test.utils import common_texts, get_tmpfile

# inp为输入语料
word2vec_path = 'word2Vec.model'
path = r"D:\machine learning\project\0_chinese_poiet\1_corpus\segment_results\jieba_set.txt"
sentences = word2vec.LineSentence(path)
model2 = word2vec.Word2Vec(sentences, size=128, window=5, min_count=5)
model2.save(word2vec_path)
model2 = word2vec.Word2Vec.load(word2vec_path)
print(model2.wv.similarity('片片', '三百'))
