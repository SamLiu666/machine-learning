import tensorflow as tf

def my_network(input):
	# 3 层网络
	w1 = tf.Variable(tf.random.uniform([784, 100], -1, 1), name='w1')
	b1 = tf.Variable(tf.zeros([100]), name='b1')
	output1 = tf.matmul(input, w1) + b1

	w2 = tf.Variable(tf.random.uniform([100, 50], -1, 1), name='w2')
	b2 = tf.Variable(tf.zeros([50]), name='b2')
	output2 = tf.matmul(output1, w2) + b2

	w3 = tf.Variable(tf.random.uniform([50, 10], -1, 1), name='w3')
	b3 = tf.Variable(tf.zeros([10]), name='b3')
	output3 = tf.matmul(output2, w3) + b3

	print("Printing names of weight parameters: ",
		w1.name, w2.name, w3.name)
	print("Printing names of weight parameters: ",
		b1.name, b2.name, b3.name)

	return output3


input1 = tf.Variable(tf.random.uniform([1000, 784], -1, 1), name='input1')
print(my_network(input1))
# print(input1)