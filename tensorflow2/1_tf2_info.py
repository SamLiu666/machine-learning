import tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets, layers, optimizers
import  os


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
print(tf.__version__)

def processes(x, y):
	""":arg
	x:[batch, 28, 28] -> [batch, 28*28]
	y:[b]"""
	x = tf.cast(x, dtype=tf.float32)/255.
	x = tf.reshape(x, [-1,28*28])
	y = tf.cast(y, dtype=tf.int32)
	y = tf.one_hot(y, depth=10) # change number class into one_hot
	return x,y