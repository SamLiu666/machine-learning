import pynlpir

def example():
# 打开API接口
    pynlpir.open()

# 其他编码方式
# pynlpir.open(encoding='big5')

    s = 'NLPIR分词系统前身为2000年发布的ICTCLAS词法分析系统，从2009年开始，为了和以前工作进行大的区隔，并推广NLPIR自然语言处理与信息检索共享平台，调整命名为NLPIR分词系统。'
    seg_tag =pynlpir.segment(s)
    segment = pynlpir.segment(s, pos_tagging=False)  # return list
    print(segment)
    pynlpir.close()   # close

import datetime
start = datetime.datetime.now()
pynlpir.open()

f = open("corpus.txt", encoding="utf-8")
lines = f.readlines()
f.close()
count = 0
output = open("pynlpir_seg1.txt", "a", encoding="utf-8")
for line in lines:
    seg_tag =pynlpir.segment(line, pos_tagging=False)
    content = " ".join(seg_tag) + "\n"
    output.write(content)
    count += 1
    if count % 1000 == 0:
        print("已处理%s 次" % count)

output.close()
end = datetime.datetime.now()
print (end-start)
pynlpir.close()   # close