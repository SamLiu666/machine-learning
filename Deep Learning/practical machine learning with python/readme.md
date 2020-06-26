[其他书籍推荐地址](https://mp.weixin.qq.com/s/xTR0SZK3bn2imM78ol6Q5g)

[book code site](https://github.com/fchollet/deep-learning-with-python-notebooks)

```python
from keras import backend as K

# Some memory clean-up
K.clear_session()

history_dict = original_hist.history
print(history_dict.keys())
# dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
```

