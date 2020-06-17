# encoding=utf-8
import jieba

def seg_example():
    """【全模式】: 我/ 来到/ 北京/ 清华/ 清华大学/ 华大/ 大学

【精确模式】: 我/ 来到/ 北京/ 清华大学

【新词识别】：他, 来到, 了, 网易, 杭研, 大厦    (此处，“杭研”并没有在词典中，但是也被Viterbi算法识别出来了)

【搜索引擎模式】： 小明, 硕士, 毕业, 于, 中国, 科学, 学院, 科学院, 中国科学院, 计算, 计算所, 后, 在, 日本, 京都, 大学, 日本京都大学, 深造"""
    jieba.enable_paddle()# 启动paddle模式。 0.40版之后开始支持，早期版本不支持
    strs=["我来到北京清华大学","乒乓球拍卖完了","中国科学技术大学"]
    for str in strs:
        seg_list = jieba.cut(str,use_paddle=True) # 使用paddle模式
        print("Paddle Mode: " + '/'.join(list(seg_list)))

    seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
    print("Full Mode: " + "/ ".join(seg_list))  # 全模式

    seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
    print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

    seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
    print(", ".join(seg_list))

    seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
    print(", ".join(seg_list))

def tga_example():
    import jieba.posseg as pseg
    words = pseg.cut("我爱北京天安门")  # jieba默认模式
    jieba.enable_paddle()  # 启动paddle模式。 0.40版之后开始支持，早期版本不支持
    # words = pseg.cut("我爱北京天安门",use_paddle=True) #paddle模式
    for word, flag in words:
        s = word + flag
        # print('%s %s' % (word, flag))
        print(s, type(s))


import datetime
import jieba.posseg as pseg
start = datetime.datetime.now()

file = open("corpus.txt", "r", encoding="utf-8")
f = file.readlines()
file.close()
# print(f[1], len(f), type(f))

count = 0
# output = open("jieba_seg.txt", "a", encoding="utf-8")
tag = open("jieba_tag.txt", "a", encoding="utf-8")

for line in f:
    # seg_list = jieba.cut(line, cut_all=False)
    # 分词并写入文件
    # seg_list = jieba.cut(line, use_paddle=True) # 使用paddle模式
    # content = ' '.join(list(seg_list))
    # output.write(content)

    # 词性标注并写入文件
    words = pseg.cut(line, use_paddle=True)
    for word, flag in words:
        s = word + " " + flag + "\n"
        tag.write(s)
    # print("Paddle Mode: " + '/'.join(list(seg_list)))
    count += 1
    if count % 1000 == 0:
        print("已处理%s 次" % count)
# output.close()
tag.close()

end = datetime.datetime.now()
print (end-start)

