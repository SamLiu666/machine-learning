import jieba.analyse
from PIL import Image,ImageSequence
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from wordcloud import WordCloud,ImageColorGenerator

font = FontProperties(fname='Songti.ttc')
bar_width = 0.5
lyric= ''

f = open(r"D:\machine learning\project\0_chinese_poiet\1_corpus\segment_results\jieba_set.txt", "r", encoding="utf-8")
result = f.readlines()

keywords = dict()
for i in result:
    keywords[i[0]]=i[1]
print(keywords)

keywords = dict()
for i in result:
    keywords[i[0]]=i[1]
print(keywords)

image= Image.open('./background.png')
graph = np.array(image)
print(graph)
wc = WordCloud(font_path='Songti.ttc',background_color='White',max_words=50,mask=graph)
wc.generate_from_frequencies(keywords)
image_color = ImageColorGenerator(graph)#设置背景图像
plt.imshow(wc)  #画图
plt.imshow(wc.recolor(color_func=image_color))  #根据背景图片着色
plt.axis("off") #不显示坐标轴
plt.show()
wc.to_file('output.png')