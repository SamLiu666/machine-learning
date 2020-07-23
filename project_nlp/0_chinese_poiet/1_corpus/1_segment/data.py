import os
import json


# 获取指定文件夹下的语料库
def file_name(path):
    # 得到文件夹下的所有文件名称
    files= os.listdir(path)
    all_path = path + "\\"
    total = []
    # 遍历文件夹
    for file in files:
        total.append(all_path+file)

    # 特殊情况，jason文件，分析后提取包含诗词的文件路径
    s = "json"
    ans = []
    for t in total:
        if s in t:
            ans.append(t)
    return ans[1:]


def read_jason(file_path):
    # 读取jason 文件
    file = open(file_path, 'r', encoding='utf-8')
    data = json.load(file)
    return data


def write_file(data):
    # 所有数据写入corpus,txt
    author, paragraphs, rhythmic = [], [], []
    for i in range(len(data)):
        info = data[i]
        author.append(info['author'])
        paragraphs.append(info['paragraphs'])
        rhythmic.append(info['rhythmic'])

    ff = open("corpus.txt", "a", encoding="utf-8")
    for i in range(len(author)):
        content = "\n" + author[i] + "\n" + " ".join(paragraphs[i]) + "\n" + rhythmic[i]
        ff.write(content)
    ff.close()


def main():
    # path 可根据需求修改
    path = "E:\chrome download\paper\corpus\chinese-poetry-master\ci"  # 文件夹目录
    ans = file_name(path)
    # print([i + '\n' for i in ans])

    for i in range(len(ans)):
        data = read_jason(ans[i])
        #     print(data)
        write_file(data)
        print("第 %s 诗词文档写入完毕" % i)


if __name__ == '__main__':
    main()