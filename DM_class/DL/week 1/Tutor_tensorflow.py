import tensorflow as tf
from keras import backend as K
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0'  # 指定使用GPU
K.clear_session()  # Some memory clean-up
print('Tensorflow2.0 version method: \n')

a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)
d = tf.multiply(a, b)
e = tf.add(b, c)
f = tf.subtract(d, e)
# f = (a*b) - (b+c)
print("f = (a*b) - (b+c) = ", f)

def exercis_2():
    g = a * a + b * f
    h = (b + c) * d + e * e * c
    print(g, h)
exercis_2()

"""Exercise 3: Create a tensor  𝑥  of shape  (3,1)  with entry values  𝑥[𝑖,0]=𝑖 ,  ∀𝑖=0,1,2  '
and another tensor  𝑦  of shape  (1,3)  with all entry values of  2 . 
Create another operation  𝑧  to perform matrix multiplication of  𝑥  and  𝑦 . 
This operation is also called outer product. 
Print the shape of  𝑧 . Also, run a querry to compute  𝑧  and print the result."""
x = tf.constant([[1],[2],[3]])
y = tf.fill(name='y', dims= [1,3], value=2)
print(x.get_shape(), y.get_shape())
z = tf.multiply(x, y)  # 多维
zz = tf.matmul(x, y) # 二维
print(z)
print(zz)

# 变量初始化
init_var = tf.random.normal(shape=[2,3], mean=0, stddev=0.1, dtype= tf.float32)
my_var=  tf.Variable(initial_value= init_var, name='x')
# init= tf.global_variables_initializer()
print("Pre-run my_var: {}".format(my_var))

print('##########################################','\n', 'Use tensorflow2 as numpy')
x = tf.constant([[1,2,3],[2,3,4],[3,3,3]])
print(x.shape)
print('转置',tf.transpose(x))

print('##########################################','\n', 'Variables')
v = tf.Variable(tf.fill(name='v', dims=[3,2], value=20))
print(v)
print('v*2', v.assign(v*2))
print("修改数字: ", v[0,1].assign(100))

print('##########################################','\n', 'Function and Graph')
def cube(x):
    return x**3
print("3^3=",cube(tf.constant(3)))

@tf.function
def tf_cube(x):
    return x**3
print(tf_cube(tf.constant(3)))

print('##########################################','\n', 'Eager Computation, AutoGraph, Tracing')
