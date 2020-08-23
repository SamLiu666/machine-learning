[其他书籍推荐地址](https://mp.weixin.qq.com/s/xTR0SZK3bn2imM78ol6Q5g)

[book code site](https://github.com/fchollet/deep-learning-with-python-notebooks)

```python
from keras import backend as K
K.clear_session()  # Some memory clean-up
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0'  # 指定使用GPU


history_dict = original_hist.history
print(history_dict.keys())
# dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

```

## VGG16

VGG16模型应用预测图片分类，生成heatmap

## NLP

```javascript
word_embeddings.py: 
输入： (samples, sequence_length)
使用预训练的glove词嵌入模型，对imdb评论分类，准确率70%，数据处理到输出
```

