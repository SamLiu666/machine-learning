数据集：IMDB

传统文本分类方法： input -> embedding -> dense -> dense -> output(sigmoid)

问题：遇到新建txt分类时，报错

```shell
2020-06-12 13:41:59.760226: W tensorflow/core/common_runtime/base_collective_executor.cc:216] BaseCollectiveExecutor::StartAbort Invalid argument: indices[0,2] = 12444 is not in [0, 10000)
         [[{{node embedding/embedding_lookup}}]]
         [[embedding/embedding_lookup/_6]]
2020-06-12 13:41:59.768590: W tensorflow/core/common_runtime/base_collective_executor.cc:216] BaseCollectiveExecutor::StartAbort Invalid argument: indices[0,2] = 12444 is not in [0, 10000)
         [[{{node embedding/embedding_lookup}}]]
Traceback (most recent call last):
  File "other.py", line 114, in <module>
    predict = model.predict(encode)
  File "D:\Anaconda\envs\tensorflow2.0\lib\site-packages\tensorflow\python\keras\engine\training.py", line 821, in predict
    use_multiprocessing=use_multiprocessing)
  File "D:\Anaconda\envs\tensorflow2.0\lib\site-packages\tensorflow\python\keras\engine\training_arrays.py", line 712, in predict
    callbacks=callbacks)
  File "D:\Anaconda\envs\tensorflow2.0\lib\site-packages\tensorflow\python\keras\engine\training_arrays.py", line 383, in model_iteration
    batch_outs = f(ins_batch)
  File "D:\Anaconda\envs\tensorflow2.0\lib\site-packages\tensorflow\python\keras\backend.py", line 3510, in __call__
    outputs = self._graph_fn(*converted_inputs)
  File "D:\Anaconda\envs\tensorflow2.0\lib\site-packages\tensorflow\python\eager\function.py", line 572, in __call__
    return self._call_flat(args)
  File "D:\Anaconda\envs\tensorflow2.0\lib\site-packages\tensorflow\python\eager\function.py", line 671, in _call_flat
    outputs = self._inference_function.call(ctx, args)
  File "D:\Anaconda\envs\tensorflow2.0\lib\site-packages\tensorflow\python\eager\function.py", line 445, in call
    ctx=ctx)
  File "D:\Anaconda\envs\tensorflow2.0\lib\site-packages\tensorflow\python\eager\execute.py", line 67, in quick_execute
    six.raise_from(core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InvalidArgumentError: 2 root error(s) found.
  (0) Invalid argument:  indices[0,2] = 12444 is not in [0, 10000)
         [[node embedding/embedding_lookup (defined at other.py:114) ]]
         [[embedding/embedding_lookup/_6]]
  (1) Invalid argument:  indices[0,2] = 12444 is not in [0, 10000)
         [[node embedding/embedding_lookup (defined at other.py:114) ]]
0 successful operations.
0 derived errors ignored. [Op:__inference_keras_scratch_graph_1033]

Function call stack:
keras_scratch_graph -> keras_scratch_graph
```



RNN 分类