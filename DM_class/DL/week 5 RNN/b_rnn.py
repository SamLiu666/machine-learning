import tensorflow as tf
import tensorflow_datasets as tfds
from collections import Counter
from tensorflow import keras
from keras import layers
import numpy

tf.random.set_seed(42)


"""rnn for sentiment analysis in tf2x"""
print("#################################################################")
print("1. implementation with tf dataset")
datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
print(datasets.keys())

train_size = info.splits["train"].num_examples
test_size = info.splits["test"].num_examples
print("Train size: {}".format(train_size))
print("Test size: {}".format(test_size))

print("###### get data")
for X_batch, y_batch in datasets["train"].batch(5).take(1):
    for review, label in zip(X_batch.numpy(), y_batch.numpy()):
        print("Review:", review)
        print("Label:", label, "= Positive" if label else "= Negative")
        print("\n")


print("#################################################################")
print("# 2 create vocabulary")
def preprocess(X_batch, y_batch):
    print("preprocessing data")
    X_batch = tf.strings.regex_replace(X_batch, "<br\s*/?>", b" ")
    X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z]", b" ")
    X_batch = tf.strings.split(X_batch)
    X_batch = X_batch[:, :100]
    return X_batch.to_tensor(default_value=b"<pad>"), y_batch


vocabulary = Counter()
for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
    for review in X_batch:
        vocabulary.update(list(review.numpy()))

vocab_size = 10000
truncated_vocabulary = [word for word, count in vocabulary.most_common()[:vocab_size]]
word_to_id = {word: index for index, word in enumerate(truncated_vocabulary)}
print(list(word_to_id.items())[0:10])

words = tf.constant(truncated_vocabulary)
word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
num_oov_buckets = 1000
table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)

def encode_words(X_batch, y_batch):
    return table.lookup(X_batch), y_batch

train_set = datasets["train"].repeat().batch(32).map(preprocess)
train_set = train_set.map(encode_words).prefetch(1)
for X_batch, y_batch in train_set.take(2):
    print(X_batch.shape)
    print(y_batch.shape)


print("#################################################################")
print("# 3. RNN model and Training set")
embed_size = 128
x = tf.keras.Input(shape=[None], dtype="int64")
print(x.shape)
h = tf.keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size)(x)
print(h.shape)
h = tf.keras.layers.GRU(embed_size, return_sequences=True)(h)
print(h.shape)
h = tf.keras.layers.GRU(64)(h)
print(h.shape)
h = tf.keras.layers.Dense(1, activation="sigmoid")(h)
rnn_model = tf.keras.models.Model(inputs = x, outputs= h)
rnn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = rnn_model.fit(train_set, steps_per_epoch=train_size // 64, epochs=3)

print("#################################################################")
print("# 3. RNN model and Training set")
from keras import backend as K
K.clear_session()
model = keras.models.Sequential()
model.add(layers.Embedding(vocab_size + num_oov_buckets, embed_size,
                           mask_zero=True, # not shown in the book
                           input_shape=[None]))
model.add(layers.GRU(128, return_sequences=True))
model.add(layers.GRU(128))
model.add(layers.Dense(1, activation="sigmoid"))
model.summary()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(train_set, steps_per_epoch=train_size // 64, epochs=3)

test_set = datasets["test"].repeat(1).batch(32).map(preprocess)
test_set = test_set.map(encode_words).prefetch(1)
model.evaluate(test_set)