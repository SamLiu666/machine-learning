import csv

header = ['MiniVGG', 'MiniVGG_aug', "ResNet"]  # 数据列名
datas = {'MiniVGG': 'Tony', 'MiniVGG_aug': 17, "ResNet": 28}
datas1 = {'MiniVGG': '李华', 'MiniVGG_aug': 21, "ResNet": 35} # 字典数据
datas2 = {'MiniVGG': 'Tony', 'MiniVGG_aug': 17, "ResNet": 20}
data = [datas,datas1,datas2]
# test.csv表示如果在当前目录下没有此文件的话，则创建一个csv文件
# a表示以“追加”的形式写入，如果是“w”的话，表示在写入之前会清空原文件中的数据
# newline是数据之间不加空行
# encoding='utf-8'表示编码格式为utf-8，如果不希望在excel中打开csv文件出现中文乱码的话，将其去掉不写也行。
# with open('test.csv', 'a', newline='', encoding='utf-8') as f:
#     writer = csv.DictWriter(f, fieldnames=header)  # 提前预览列名，当下面代码写入数据时，会将其一一对应。
#     writer.writeheader()  # 写入列名
#     writer.writerows(datas)  # 写入数据
#     writer.writerows(datas1)  # 写入数据
#     writer.writerows(datas2)  # 写入数据
#     f.close()

with open('model.txt', 'a', newline='', encoding='utf-8') as f:
    for d in data:
        for i,j in d.items():
            f.writelines((i+"   "+str(j) + "    "))
        f.write("\n")
    # f.writelines(str(datas))
    # f.writelines(str(datas2))
    f.close()