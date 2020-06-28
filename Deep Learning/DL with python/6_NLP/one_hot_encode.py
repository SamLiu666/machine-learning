import keras
import numpy as np
import string
from keras.preprocessing.text import Tokenizer

def word_one_hot(samples):
    # samples = ['I love playing basketball', 'I like kicking soccer']
    token_index = {}
    for sample in samples:
        for word in sample.split():
            if word not in token_index:
                # 索引从1开始
                token_index[word] = len(token_index) + 1

    # 每个词的最大长度
    max_length = 10
    results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = token_index.get(word)
            results[i, j, index] = 1
    return results

def charactor(samples):
    characters = string.printable  # All printable ASCII characters.
    token_index = dict(zip(characters, range(1, len(characters) + 1)))

    max_length = 50
    results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
    for i, sample in enumerate(samples):
        for j, character in enumerate(sample[:max_length]):
            index = token_index.get(character)
            results[i, j, index] = 1.

    print(results)

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
# 1000个常用词
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# We will store our words as vectors of size 1000.
# Note that if you have close to 1000 words (or more)
# you will start seeing many hash collisions, which
# will decrease the accuracy of this encoding method.
dimensionality = 1000
max_length = 10

results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        # Hash the word into a "random" integer index
        # that is between 0 and 1000
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1.