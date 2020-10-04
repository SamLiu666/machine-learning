import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
tfds.disable_progress_bar()


def show_plot(history):
    history_dict = history.history

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss=history_dict['loss']
    val_loss=history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12,9))
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12,9))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim((0.5,1))
    plt.show()

# 使用嵌入词向量
embedding_layer = tf.keras.layers.Embedding(1000, 5)
result = embedding_layer(tf.constant([[0,1,2],[3,4,5]]))
print("result shape: ", tf.constant([1,2,3]), result.shape)

(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k',
    split = (tfds.Split.TRAIN, tfds.Split.TEST),
    with_info=True, as_supervised=True)

encoder = info.features['text'].encoder
print(encoder.subwords[:20])

# 将使用 padded_batch 方法来标准化评论的长度。
train_batches = train_data.shuffle(1000)
test_batches = test_data.shuffle(1000).padded_batch(batch_size=10, padded_shapes=None)

train_batch, train_labels = next(iter(train_batches))
print("Dataset shape: ", train_batch[0].shape, test_batches[0].shape)
# 创建一个模型
embedding_dim = 32
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, embedding_dim),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1)
])
model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(
    train_batches,
    epochs=10,
    validation_data=test_batches, validation_steps=20)

show_plot(history)

# 检索学习的嵌入向量
weights = model.layers[0].get_weights()[0]

import io

encoder = info.features['text'].encoder

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for num, word in enumerate(encoder.subwords):
    vec = weights[num+1] # skip 0, it's padding.
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()
