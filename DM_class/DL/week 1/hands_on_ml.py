import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import tensorflow as tf

def chapter_3():
    iris = load_iris()
    # petal length, width
    X = iris.data[:, (2,3)]
    y = (iris.target == 0).astype(np.int)
    per_clf = Perceptron(random_state=42)
    print(per_clf.fit(X, y))

    y_pred = per_clf.predict([[2,0.5]])
    print(y_pred)
    print(X.shape)

    with tf.device("/gpu:0"):
        i = tf.Variable(3)
        print(i)


cluster_spec = tf.train.ClusterSpec({
"ps": [
"machine-a.example.com:2221", # /job:ps/task:0
],
"worker": [
"machine-a.example.com:2222", # /job:worker/task:0
"machine-b.example.com:2222", # /job:worker/task:1
]})
server = tf.train.Server(cluster_spec, job_name='worker', task_index=0)