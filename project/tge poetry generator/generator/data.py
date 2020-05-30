path = r'D:\machine learning\project\tge poetry generator\data\cor.txt'
f = open(path, 'r', encoding='utf-8')

corpus = f.read()
# print(corpus[:100])

corpus = corpus.replace('\n', ' ').replace('\r', ' ')
# print(corpus[:100])
print(len(corpus))
corpus = corpus[0:20000]
# print(corpus)

# 字符-》整数；整数-》字符
idx2char = list(set(corpus))
char2idx = dict([(char, i) for i, char in enumerate(idx2char)])
index = dict([(i, char) for char, i in char2idx.items()])
# print(len(idx2char))
# print(char2idx)
print(index)

# 将训练数据集中每个字符转化为索引，并打印前20个字符及其对应的索引
example = [char2idx[char] for char in corpus]
sample = example[:30]
print("chars: ", "".join(idx2char[idx] for idx in sample))
print("index: ", sample)
