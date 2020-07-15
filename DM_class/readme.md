tensorboard:

```python
# 使用tf 1x
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tensorboard --logdir logs
http://localhost:6006  # 输入网址 可视化
```
