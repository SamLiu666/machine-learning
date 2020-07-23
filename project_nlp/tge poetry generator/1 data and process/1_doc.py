import os

"""This part is dealing with documents"""


def file_rename(file_path):
    """
    It extracts the names of all the documents in the folder,
    and change the document name in "number.txt" format for convience.
    :param file_path: the name of documents in the folder
    :return:  none
    """
    # file_path = r'E:\chrome download\paper\corpus\tge'
    file_list = os.listdir(file_path)
    for i, file in enumerate(file_list):
        new = os.path.join(file_path,str(i)+'.txt')
        old = os.path.join(file_path, file)
        os.rename(old, new)


def build_corpus(file_path=None):
    '''
    transport messages all the txt files in the one
    :param file_path:  fold of txt files
    :return:  nothing
    '''
    file_path = r'E:\chrome download\paper\corpus\tge'
    file_list = os.listdir(file_path)

    res = ""  # 字符串连接，也可选用列表

    for file in file_list:
        path = os.path.join(file_path, file)
        #     r = open(path, "r")
        #     new = os.path.join(file_path, 't.txt')
        #     w = open(new)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            f.close()
        append = "%s" % (content)
        res += append

    new = os.path.join(file_path, 't.txt')  # 目标文件
    with open(new, "w", encoding="utf-8") as ff:
        ff.write(res)
        ff.close()

    print(len(res))
    print("success")

if __name__ == '__main__':
    build_corpus()