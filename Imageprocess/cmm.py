# Author LGF
# GreatTime 2020/5/19
# Description: simple introduction of the code
import os

# 输入文件位置，注意实现双斜杠
path = '../../test03/FER_Test/data/val/1/'
# 该文件夹下所有的文件
filelist = os.listdir(path)
count = 1
for file in filelist:
    print(file)
for file in filelist:  # 遍历所有文件
    Olddir = os.path.join(path, file)  # 原来的文件路径
    filename = os.path.splitext(file)[0]  # 文件名
    # filetype = os.path.splitext(file)[1]  # 文件扩展名
    Newdir = os.path.join(path, "2"+str(count).zfill(3) + ".jpg")  # 用字符串函数zfill 以0补全所需位数
    os.rename(Olddir, Newdir)  # 重命名
    count += 1
