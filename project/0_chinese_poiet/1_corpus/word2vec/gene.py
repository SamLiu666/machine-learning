import tensorflow as tf

import numpy as np
import os
import time

checkpoint_dir = "./training_checkpoints/ckpt_10"
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()