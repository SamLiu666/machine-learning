tensorboard:

```python
# 使用tf 1x
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tensorboard --logdir logs
```
