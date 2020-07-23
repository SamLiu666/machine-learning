# 宋词创作

## 1 获取数据 

data.py： 读取文件夹下所有（21份）宋词文件，并写入到corpus.txt 文件中

```jason
[ 源文件格式
  {
    "author": "石孝友", 
    "paragraphs": [
      "扁舟破浪鸣双橹。", 
      "岁晚客心分万绪。", 
      "香红漠漠落梅村，愁碧萋萋芳草渡。", 
      "汉皋佩失诚相误。", 
      "楚峡云归无觅处。", 
      "一天明月缺还圆，千里伴人来又去。"
    ], 
    "rhythmic": "玉楼春"
  }
]
```

保存至corpus.txt 

```python
和岘  # 作者名字
气和玉烛，睿化著鸿明。 缇管一阳生。 郊禋盛礼燔柴毕，旋轸凤凰城。 森罗仪卫振华缨。 载路溢欢声。 皇图大业超前古，垂象泰阶平。 岁时丰衍，九土乐升平。 睹寰海澄清。 道高尧舜垂衣治，日月并文明。 嘉禾甘露登歌荐，云物焕祥经。 兢兢惕惕持谦德，未许禅云亭。  # 词内容
导引  # 词牌名
```



## 2 分词

分词工具比较：https://blog.csdn.net/shuihupo/article/details/81540433

THULAC：清华大学分词工具  https://github.com/thunlp/THULAC-Python

PyNLPIR:中科院分词 https://github.com/tsroten/pynlpir

jieba：“结巴”中文分词：做最好的 Python 中文分词组件 https://github.com/fxsjy/jieba

```python
# 分词时间
import datetime
start = datetime.datetime.now()
# 分词形式
end = datetime.datetime.now()
print (end-start)

echo %time%
cmd command
echo %time
```



| 分词工具 | 问题                                                         | 评价                 | 运行时间                 |
| -------- | :----------------------------------------------------------- | -------------------- | ------------------------ |
| THULAC   | 明明保存为utf-8 的txt 文件却无法分词，显示gbk错误； 另存为ASCII码之后可以加载模型并分词issue，最后结果还是失败。 回复很少，社区内容少 | 更新过于久远，不推荐 | 3min44                   |
| PyNLPIR  | 简单好用，词性标注和分词操作简单                             | 可用                 | 13.55分词                |
| LTP 4    | pip 直接安装失败，clone+setup.py 安装失败，                  |                      |                          |
| HanLp    | 安装失败，不友好                                             |                      |                          |
| jieba    | 安装操作简单，分词和词性标注已使用，其他功能待使用           | 简单易用             | 12.6161分词 4:07词性标注 |

```python
# 分词效果
jieba：气和玉烛 ， 睿化 著鸿明 。   缇 管 一阳生 。   郊 禋 盛礼 燔 柴 毕 ， 旋轸 凤凰 城 。   森罗 仪卫 振华 缨 。
```

