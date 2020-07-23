import thulac

def examples():
    thu1 = thulac.thulac()  # 默认模式
    text = thu1.cut("我爱北京天安门", text=True)  # 进行一句话分词
    print(text)

    # 代码示例2
    # 只进行分词，不进行词性标注
    thu1 = thulac.thulac(seg_only=True)
    # 对input.txt文件内容进行分词，输出到output.txt
    thu1.cut_f("input.txt", "output.txt")

    """:arg
    thulac(user_dict=None, model_path=None, T2S=False, seg_only=False, filt=False, deli='_') 
     user_dict	      	设置用户词典，用户词典中的词会被打上uw标签。词典中每一个词一行，UTF8编码
    T2S					默认False, 是否将句子从繁体转化为简体
    seg_only	   		默认False, 时候只进行分词，不进行词性标注
    filt		   		默认False, 是否使用过滤器去除一些没有意义的词语，例如“可以”。
    model_path	 	    设置模型文件所在文件夹，默认为models/
    deli	 	      	默认为‘_’, 设置词与词性之间的分隔符
    初始化程序，进行自定义设置
    
    python -m thulac input.txt output.txt
#从input.txt读入，并将分词和词性标注结果输出到ouptut.txt中

#如果只需要分词功能，可在增加参数"seg_only" 
python -m thulac input.txt output.txt seg_only
    """
import datetime
start = datetime.datetime.now()

thu1 = thulac.thulac(seg_only=True)
f = open("corpus.txt", encoding="utf-8")
thu1.cut_f("corpus.txt", "tsing_segment.txt")
f.close()
end = datetime.datetime.now()
print (end-start)
