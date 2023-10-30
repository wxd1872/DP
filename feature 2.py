# -*- coding: utf-8 -*-
"""
@Time : 2023/4/20
@Author : 王向东
@Email : wxd1872@163.com
@File : feature 2.py
@Project : feature_select.py
"""
import os
import numpy as np
import pandas as pd
import re
import glob
import os
import numpy as np
import pandas as pd


# 获得能带投影比例新的化学式
# rootdir = 'E:\py\poscar'
rootdir = 'D:\桌面\态密度处理\gnomag_dos'
# rootdir = 'D:\桌面\态密度处理\gtest'
filepath = os.listdir(rootdir)
# print(filepath)
composition = []
doss = []
# 初始化计数器变量
count = 0

for i in range(0, len(filepath)):
    print("*" * 100)
    print(i)
    filepathfolder = os.path.join(rootdir, filepath[i])
    # print(filepathfolder)
    folder_path = filepathfolder
    print(filepath[i])
    filename = filepath[i]
    identifier = filename.split("-")[-1].split(".")[0]
    # for filename in filenames:
    # 通过正则表达式匹配文件名中的化学式部分
    # formula = re.search(r"_([A-Z][a-z]*\d*)+", filepath[i]).group(0)[1:]
    # 打印匹配到的化学式
    print(identifier)
    non_digits = re.findall(r'[A-Za-z]+', identifier)
    # 获取文件夹中符合要求的文件txt
    extension = '.txt'
    files = [file for file in os.listdir(folder_path) if file.endswith(extension)]

    # 逐个读取文件
    for file in files:
        file_path = os.path.join(folder_path, file)
    # print(file_path)
    with open(file_path, 'r') as f:
        # 使用readlines()函数逐行读取文件
        lines = f.readlines()
        # 获取第2行，并按空格分割成单个数值
        my_list = lines[1]
        my_list1 = lines[4]
        # s = 'The distance between CBM and fermilevel is 3.83403600'
        pattern = r'\d+\.\d+'
        match = re.search(pattern, my_list1)
        if match:
            sub_str = match.group()
            dos = float(sub_str) + 0.2
            print(dos)
        print(sub_str)
        # match = re.search(r'with value\s+(-?\d+\.\d+)', my_list)
        # # print(match)
        # if match:
        #     value = match.group(1)
        #     dos = float(value) + 0.2
        #     # print(type(value))
        #     print(value)
            # print(dos)

    # 获取文件夹中的dat文件
    dat_files = glob.glob(os.path.join(folder_path, "*.dat"))
    # print("#" * 100)
    for dat_file in dat_files:
        with open(dat_file, 'r') as f:
            data = f.read()
            # 除去前两行数据
            # 使用字符串的split()方法将其转换为二维列表
            data_lines = data.strip().split('\n')[2:]
            # 将字符串列表转换为二维列表
            list_2d = [row.split() for row in data_lines]
            # 将二维列表转换为Numpy数组
            arr = np.array(list_2d, dtype=float)
            # 输出Numpy数组
            # print(arr)
            # 获取第一列
            first_col = arr[:, 0]
            # print(type(first_col))
            # 将每个元素减去导带能量
            first_col = first_col - dos
            # 找到绝对值最小的元素及其所在行的索引
            min_value = np.min(np.abs(first_col))
            min_index = np.where(np.abs(first_col) == min_value)[0][0]
            print("第一列数据中绝对值最小的数为：", min_value)
            print("该数所在的行索引为：", min_index)
            va = arr[min_index]
            print(va)
            val = arr[min_index][1:-1]
            val_list = val.tolist()
            formula = ''.join([non_digits[i] + str(val[i]) for i in range(len(val))])
            print(formula)
            print(val_list)
            print(type(val_list))
            valley = arr[min_index]
            # print(valley)
            doss.append(val_list)
            composition.append(formula)
            # count += 1
            # print("迭代次数", count)
# print(doss)
# print(type(doss))
# print(len(filepath))
print(len(doss))
df = pd.DataFrame({'composition': filepath, 'dos_0,2': doss, 'newcomposition' : composition})

# # 将第一列和最后一列除去，保留中间列
# sub_data = [row[1:-1] for row in data]
# # 打印结果
# print(sub_data)
print(df)
# 将数据框保存为CSV文件
df.to_csv('my_data2.csv', index=False)




