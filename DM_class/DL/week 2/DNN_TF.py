import numpy as np
import random


def sigmoid(x):
    return 1.0/(1+np.exp(-x))


def DNN_numpy():
    # DNN forward with Numpy
    x = np.array([[1],[-2],[1]])
    w = np.random.random((4, 3))
    b = np.ones((4,1))

    h = np.dot(w, x) + b
    y = sigmoid(x)
    print("forward: \n", h)

    # DNN backward with numpy
    h_grad = y*(1-y)
    d = np.diag(h_grad[:, 0])
    print("Chain Rule:\n",d)
    derivative = d.dot(x)
    print("dh/dx=\n", derivative)


DNN_numpy()