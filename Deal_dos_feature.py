# -*- coding: utf-8 -*-
"""
@Time : 2023/4/16
@Author : 王向东
@Email : wxd1872@163.com
@File : Deal_dos_feature.py
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

#获得态密度除以体积特征
# rootdir = 'E:\py\poscar'
rootdir = 'D:\桌面\态密度处理\gnomag_dos'
# rootdir = 'D:\桌面\态密度处理\gtest'
filepath = os.listdir(rootdir)
# print(filepath)
# composition = []
doss = []
# 初始化计数器变量
count = 0

for i in range(0, len(filepath)):
    print("*" * 100)
    print(i)
    filepathfolder = os.path.join(rootdir, filepath[i])
    print(filepathfolder)
    folder_path = filepathfolder
    # print(filepath)
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
            # 将每个元素减去10
            first_col = first_col - dos
            # 找到绝对值最小的元素及其所在行的索引
            min_value = np.min(np.abs(first_col))
            min_index = np.where(np.abs(first_col) == min_value)[0][0]
            print("第一列数据中绝对值最小的数为：", min_value)
            print("该数所在的行索引为：", min_index)
            val = arr[min_index][-1]
            print(val)
            valley = arr[min_index]
            # print(valley)
            doss.append(val)
            count += 1
            print("迭代次数", count)
print(doss)
print(len(filepath))
print(len(doss))
df = pd.DataFrame({'composition': filepath, 'dos_0,2': doss})
print(df)
# 将数据框保存为CSV文件
df.to_csv('my_data17.csv', index=False)







































    # # 获取文件夹中符合要求的文件txt
    # extension = '.txt'
    # files = [file for file in os.listdir(folder_path) if file.endswith(extension)]
    # # 逐个读取文件
    # for file in files:
    #     file_path = os.path.join(folder_path, file)
    # print(file_path)
    # with open(file_path, 'r') as f:
    #     # 使用readlines()函数逐行读取文件
    #     lines = f.readlines()
    #     # 获取第2行，并按空格分割成单个数值
    #     my_list = lines[1]
    #     match = re.search(r'with value\s+(-?\d+\.\d+)', my_list)
    #     # print(match)
    #     if match:
    #         value = match.group(1)
    #         print(value)
























        # # 记录最小值及其索引位置
            # min_index = -1
            # min_value = float("inf")
            #
            # # 将字符串数据分割成行，并逐行处理
            # for line in data.strip().split('\n'):
            #
            #     # 将行数据分割成各列数据，并获取第一列数据
            #     cols = line.split()
            #     first_col = float(cols[0])
            #     # 对第一列数据进行遍历处理
            #     # print("#"*100)
            #     # print(first_col)
            #
            #
            #     val = abs(first_col - 10)
            #     if val < min_value:
            #         min_value = val
            #         min_index = i
            #     print(first_col)
            #
            # # print(data)
            # # print("*"*100)
            # # print(type(data))

            # print(line)
            # 获取第2个数字
            # number = line[10]
            # print(number)


        # def gaptxt(self, filename='gap.txt'):
        #     """analysis gap.txt"""
        #     try:
        #         if os.path.isfile(filename):
        #             with open(filename, 'r+') as fgap:  # gap.txt
        #             lines_fgap = fgap.read()
        #     result_gap = re.search(r'The gap is\s+(.*?)\n', lines_fgap)
        #     if result_gap:
        #         self.gap = float(result_gap.group(1))
        #     result_vbm = re.search(
        #         r'The k-point of VBM is (.*?), with value\s+(.*?) , band no.\s+(\d+)\n', lines_fgap)
        #     if result_vbm:
        #         self.k_vbm = result_vbm.group(1)
        #     self.value_vbm = float(result_vbm.group(2))
        #     self.NO_vb = int(result_vbm.group(3))
        #     result_cbm = re.search(
        #         r'The k-point of CBM is (.*?), with value\s+(.*?) , band no.\s+(\d+)\n', lines_fgap)

    #
    #     if result_cbm:
    #         self.k_cbm = result_cbm.group(1)
    #     self.value_cbm = float(result_cbm.group(2))
    #     self.NO_cb = int(result_cbm.group(3))
    #     result_fermi = re.search(r'The Fermilevel is\s+(.*?)\n', lines_fgap)
    # if result_fermi:
    #     self.fermi = float(result_fermi.group(1))

    # data = pd.read_csv(file_path, sep='\t', header=None)
        # # print(data)
        # print(type(data))
        #
        #
        #     # 获取第3行，并按空格分割成单个数值
        #     line = lines[2].strip().split(' ')
        #
        #     # 获取第2个数字
        #     number = line[1]
        # value = data.iloc[1, 1]
        # print(value)
    # print("*" * 100)
    # 在此处添加你的处理代码


# # 列出文件夹下所有的目录与文件
# for root, dirs, files in os.walk(rootdir):
#     for dir in dirs:
#         subdir_path = os.path.join(root, dir)
#
#         extension = '.txt'
#         # 获取文件夹中符合要求的文件
#         files = [file for file in os.listdir(subdir_path) if file.endswith(extension)]
#
#         # 逐个读取文件
#         for file in files:
#             file_path = os.path.join(folder_path, file)
#             data = pd.read_csv(file_path, sep='\t', header=None)
#
#         # for filename in os.listdir(subdir_path):
#         #     file_path = os.path.join(subdir_path, filename)
#         #     print(file_path)
#         #     print("*"*100)
#         #     if os.path.isfile(file_path):
#         #         print(filename)
#
#
# # 声明路径和要查找的文件类型
# folder_path = 'path/to/folder'
#
#
# # 获取文件夹中符合要求的文件
# files = [file for file in os.listdir(folder_path) if file.endswith(extension)]
#
# # 逐个读取文件
# for file in files:
#     file_path = os.path.join(folder_path, file)
#     data = pd.read_csv(file_path, sep='\t', header=None)
#     # 在此处添加你的处理代码


# composition = []
# for i in range(0, len(filepath)):
#     file = os.path.join(rootdir, filepath[i])
#     print(file)

    # with open(file, 'r') as fpos:
    #     pos = fpos.readlines()

# rootdir = 'D:\桌面\态密度处理\gnomag_dos'
# filepath = os.listdir(rootdir)
# # 列出文件夹下所有的目录与文件
# print(filepath)
# for root, dirs, files in os.walk(filepath):
#     for dir in dirs:
#         subdir_path = os.path.join(root, dir)
#         for filename in os.listdir(subdir_path):
#             file_path = os.path.join(subdir_path, filename)
#             if os.path.isfile(file_path):
#                 print(filename)



# for root, dirs, files in os.walk(filepath):
#     for root, dirs, files in os.walk(folder_path):
#         for dir in dirs:
#             subdir_path = os.path.join(root, dir)
#             for filename in os.listdir(subdir_path):
#                 file_path = os.path.join(subdir_path, filename)
#                 if os.path.isfile(file_path):
#                     print(filename)
    # print(dirs)
    # print(root)
# for filename in os.listdir(filepath):
#     file_path = os.path.join(filepath, filename)
#     if os.path.isfile(file_path):
#         print(filename)



# composition = []
# for i in range(0, len(filepath)):
#     file = os.path.join(rootdir, filepath[i])
#     with open(file, 'r') as fpos:
#         pos = fpos.readlines()



# import os
#
# #遍历文件夹
# def iter_files(rootDir):
#     #遍历根目录
#     for root, dirs, files in os.walk(rootDir):
#         for file in files:
#             file_name = os.path.join(root,file)
#             print(file_name)

