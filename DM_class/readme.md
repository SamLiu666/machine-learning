data -> model -> accuracy  -> save -> predict 

tensorboard:

```python
# 使用tf 1x
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from keras import backend as K
K.clear_session()  # Some memory clean-up

# 模型保存图片
from keras.utils import plot_model
plot_model(model, to_file='sentiment_analysis/my_cnn.png', show_shapes=True, show_layer_names=False)

tensorboard --logdir logs
http://localhost:6006  # 输入网址 可视化
        
# 将整个模型保存为HDF5文件
model.save('my_model.h5')
# 重新创建完全相同的模型，包括其权重和优化程序
new_model = keras.models.load_model('my_model.h5')
# 显示网络结构
new_model.summary()

```

保存模型参考：[https://www.tensorflow.org/tutorials/keras/save_and_load#%E8%BF%99%E4%BA%9B%E6%96%87%E4%BB%B6%E6%98%AF%E4%BB%80%E4%B9%88%EF%BC%9F](https://www.tensorflow.org/tutorials/keras/save_and_load#这些文件是什么？)

1. checkpoint save 保存了每一个回调点，调用的话可以直接使用参数

2. 将模型保存为HDF5文件这项技术可以保存一切:

- 权重
- 模型配置(结构)
- 优化器配置

Keras 通过检查网络结构来保存模型。目前，它无法保存 Tensorflow 优化器（调用自 [`tf.train`](https://www.tensorflow.org/api_docs/python/tf/train)）。使用这些的时候，您需要在加载后重新编译模型，否则您将失去优化器的状态。