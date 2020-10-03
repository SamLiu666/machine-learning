import tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets, layers, optimizers
import  os

print("########## Run")
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

(x,y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
db = tf.data.Dataset.from_tensor_slices((x,y))
db = db.map(processes).shuffle(10000).batch(128)
db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.map(processes).batch(128)
print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test)
# change dataset into tensor
train_data = tf.data.Dataset.from_tensor_slices((x,y))
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# print(len(train_data))

# shuffle the dataset
train_db = train_data.shuffle(60000).batch(128).map(processes).repeat(30)
test_db = test_data.shuffle(10000).batch(128).map(processes)
# print(len(train_db))
# iterator
x,y = next(iter(train_db))
print('train sample:', x.shape, y.shape)

def self_mlp_build():
	# learning rate
	lr = 1e-3
	# 784 => 392
	w1, b1 = tf.Variable(tf.random.truncated_normal([784, 392], stddev=0.1)), tf.Variable(tf.zeros([392]))
	# 392 => 256
	w2, b2 = tf.Variable(tf.random.truncated_normal([392, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
	# 256 => 10
	w3, b3 = tf.Variable(tf.random.truncated_normal([256, 10], stddev=0.1)), tf.Variable(tf.zeros([10]))

	# train the forward
	for step,(x,y) in enumerate(train_db):
		with tf.GradientTape() as tape:
			# trainning
			h1 = tf.nn.relu(tf.matmul(x, w1)+b1) # layer1: 784 => 392
			h2 = tf.nn.relu(tf.matmul(h1,w2)+b2) # layer2: 392 => 256
			out = tf.nn.relu(tf.matmul(h2,w3)+b3) # layer3: 256 => [b,10]

			# compute loss
			loss = tf.square(y-out) # l2_loss [b,10]
			loss = tf.reduce_mean(loss, axis=1) # [b]
			loss = tf.reduce_mean(loss) # [b] -> scalar

		# compute gradient
		grads = tape.gradient(loss, [w1,b1,w2,b2,w3,b3])
		for p,g in zip([w1,b1,w2,b2,w3,b3], grads):
			p.assign_sub(lr * g)

		# print
		if step % 100 == 0:
			print(step, 'loss:', float(loss))

		# evaluate
		if step % 500 == 0:
			total, total_correct = 0., 0

			for step, (x, y) in enumerate(test_db):
				# layer1.
				h1 = x @ w1 + b1
				h1 = tf.nn.relu(h1)
				# layer2
				h2 = h1 @ w2 + b2
				h2 = tf.nn.relu(h2)
				# output
				out = h2 @ w3 + b3
				# [b, 10] => [b]
				pred = tf.argmax(out, axis=1)
				# convert one_hot y to number y
				y = tf.argmax(y, axis=1)
				# bool type
				correct = tf.equal(pred, y)
				# bool tensor => int tensor => numpy
				total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
				total += x.shape[0]

			print(step, 'Evaluate Acc:', total_correct / total)


model = tf.keras.Sequential([
	tf.keras.layers.Dense(256, activation=tf.nn.relu),
	layers.Dense(128, activation=tf.nn.relu),  # [b, 256] => [b, 128]
	layers.Dense(64, activation=tf.nn.relu),  # [b, 128] => [b, 64]
	layers.Dense(32, activation=tf.nn.relu),  # [b, 64] => [b, 32]
	layers.Dense(10)  # [b, 32] => [b, 10], 330 = 32*10 + 10
])
model.build(input_shape=[None, 28*28])
model.summary()
optimizer = tf.keras.optimizers.Adam(lr=1e-3)
for epoch in range(30):

	for step, (x, y) in enumerate(db):

		#x: [b, 28, 28] => [b, 784]
		# y: [b]
		# x = tf.reshape(x, [-1, 28 * 28])

		with tf.GradientTape() as tape:
			# [b, 784] => [b, 10]
			logits = model(x)
			loss_mse = tf.reduce_mean(tf.losses.MSE(y, logits))
			loss_ce = tf.losses.categorical_crossentropy(y, logits, from_logits=True)
			loss_ce = tf.reduce_mean(loss_ce)

		grads = tape.gradient(loss_ce, model.trainable_variables)
		optimizer.apply_gradients(zip(grads, model.trainable_variables))

		if step % 100 == 0:
			print(epoch, step, 'loss:', float(loss_ce), float(loss_mse))
	# test
	total_correct = 0
	total_num = 0
	for x, y in db_test:
		# x: [b, 28, 28] => [b, 784]
		# y: [b]
		x = tf.reshape(x, [-1, 28 * 28])
		# [b, 10]
		logits = model(x)
		# logits => prob, [b, 10]
		prob = tf.nn.softmax(logits, axis=1)
		# [b, 10] => [b], int64
		pred = tf.argmax(prob, axis=1)
		pred = tf.cast(pred, dtype=tf.int32) # pred:[b]

		y = tf.argmax(y, axis=1)  # y: [b]
		correct = tf.equal(logits, y)  # correct: [b], True: equal, False: not equal
		correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

		total_correct += int(correct)
		total_num += x.shape[0]

	acc = total_correct / total_num
	print(epoch, 'test acc:', acc)