import tensorflow as tf
import numpy as np
import os, time,matplotlib
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import layers
K.clear_session()

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

print("#################################################################")
print("1. manual RNN")
X0 = np.array([[0.0, 1.0, -2.0],
               [-3.0, 4.0, 5.0],
               [6.0, 7.0, -8.0],
               [6.0, -1.0, 2.0]], dtype= np.float32) # t = 0
X1 = np.array([[9.0, 8.0, 7.0],
               [0.0, 0.0, 0.0],
               [6.0, 5.0, 4.0],
               [1.0, 2.0, 3.0]], dtype= np.float32) # t = 1


U = tf.Variable(tf.random.normal(shape=[3,5], dtype=tf.float32))
W = tf.Variable(tf.random.normal(shape=[5,5], dtype=tf.float32))
#b = tf.Variable(tf.zeros([1, 5], dtype=tf.float32))
b = tf.Variable(tf.random.normal([1, 5], dtype=tf.float32))

h0 = tf.tanh(tf.matmul(X0, U) + b)
h1 = tf.tanh(tf.matmul(X1, U) + tf.matmul(h0, W) + b)

print("h0= {}".format(h0.numpy()))
print("h1= {}".format(h1.numpy()))


print("#################################################################")
print("2. Recurrent cells in TF keras")
print(""" standard RNN, LSTM, or GRU cell) 
has the shape """)

inputs = np.random.random([32,10,8]).astype(np.float32)
simple_rnn = tf.keras.layers.SimpleRNN(4)
output = simple_rnn(inputs)
print(output.shape)

simple_rnn = layers.SimpleRNN(4, return_sequences=True, return_state=True)
whole_sequence, final_state = simple_rnn(inputs)
print("whole_sequence, final_state: ", whole_sequence.shape, final_state.shape, sep="\n")

simple_rnn = layers.SimpleRNN(4, return_sequences=False, return_state=True)
whole_sequence, final_state = simple_rnn(inputs)
print("whole_sequence, final_state: ", whole_sequence.shape, final_state.shape, sep="\n")

print("#################################################################")
print("3. LSTM cells in TF keras")
lstm = layers.LSTM(4)
output = lstm(inputs)
print("LSTM : ",output.shape)

lstm = tf.keras.layers.LSTM(4, return_sequences=True, return_state=True)
whole_seq_output, final_memory_state, final_hidden_state = lstm(inputs)
print(whole_seq_output.shape)    #h = [h1, h2,..., hL]
print(final_memory_state.shape)  #cL
print(final_hidden_state.shape)   #hL

print("#################################################################")
print("4. GRU cells in TF keras")
gru = layers.GRU(4)
output = gru(inputs)
print(output.shape)

gru = tf.keras.layers.GRU(4, return_sequences=True, return_state=True)
whole_sequence_output, final_hidden_state = gru(inputs)
print(whole_sequence_output.shape)
print(final_hidden_state.shape)