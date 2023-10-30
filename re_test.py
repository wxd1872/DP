# -*- coding: utf-8 -*-
"""
@Time : 2023/4/18
@Author : 王向东
@Email : wxd1872@163.com
@File : re_test.py
@Project : feature_select.py
"""

# 将态密度/体积
import pandas as pd
# 读取CSV文件
df1 = pd.read_csv('D:\桌面\态密度处理\g222.csv')
df2 = pd.read_csv('D:\桌面\态密度处理\g333.csv')

# 按照第一列'A'合并数据表
merged_df = pd.merge(df1, df2, on='composition', how='outer')
print(merged_df)

# 计算新列
merged_df['D'] = merged_df.apply(lambda x: x['dos_0,2'] / x['volume'] * 1000, axis=1)
print(merged_df)
merged_df.to_csv('feature1.csv', index=False)















# import re
#
# string = "The k-point of CBM is (0.4666667,0.4666667,0.0000000), with value 10.19617300, band no. 14"
# match = re.search(r'with value\s+(\d+\.\d+)', string)
# if match:
#     value = match.group(1)
#     print(value)

# import os
#
# root_path = 'D:\桌面\态密度处理\gnomag_dos'
#
# # 遍历根目录下的所有子目录
# for dirpath, dirnames, filenames in os.walk(root_path):
#     # 如果该目录中没有任何.dat文件，则认为该目录不包含dat文件
#     if not any(filename.endswith('.dat') for filename in filenames):
#         print("Directory without .dat files:", dirpath)











# import os
#
# root_path = 'D:\桌面\态密度处理\gnomag_dos'
#
# # 遍历根目录下的所有子目录
# for dirpath, dirnames, filenames in os.walk(root_path):
#     # 如果该目录没有文件和文件夹，则认为这是一个空目录
#     if len(os.listdir(dirpath)) == 0:
#         print("Empty directory:", dirpath)


# import numpy as np
#
# # 将数据存储在字符串变量中
# data_str = '''26.6696   0.1239   0.4282   0.1799   0.9596
#             26.6901   0.1264   0.4494   0.1863   1.0132
#             26.7105   0.1254   0.4718   0.1838   1.0571
#             26.7309   0.1207   0.4891   0.1745   1.0779'''
#
# # 使用字符串的split()方法将其转换为二维列表
# data_list = [row.split() for row in data_str.split('\n')]
#
# # 将列表转换为Numpy数组
# data_np = np.array(data_list, dtype=float)
#
#
#
#
# # 输出Numpy数组
# print(data_np)

# print(data_lines)
# data_list = [row.split() for row in data_lines.split('\n')]
# print(data_list)
# # 将列表转换为Numpy数组
# data_np = np.array(data_list, dtype=float)
# # 删除前两行数据
# data_np = data_np[2:, :]

# # 输出Numpy数组
# print(data_np)


#
# data_list = [row.split() for row in data_str.split('\n')]
# print(type(data_lines))
#
# data = np.array(data_lines)
# print(data)


# # 获取第一列
# first_col = data[:, 0]
# # 将每个元素减去10
# first_col = first_col - 10
# # 找到绝对值最小的元素及其所在行的索引
# min_value = np.min(np.abs(first_col))
# min_index = np.where(np.abs(first_col) == min_value)[0][0]
# print("第一列数据中绝对值最小的数为：", min_value)
# print("该数所在的行索引为：", min_index)

# print(type(data))
# for i, line in enumerate(data_lines):
#     # 将行数据分割成各列数据，并获取第一列数据
#     cols = line.split()
#     first_col = float(cols[0])
#     # print(type(first_col))


# # 记录最小值及其索引位置
# min_index = -1
# min_value = float("inf")
# # # 将字符串数据分割成行，并逐行处理
# # for i, line in enumerate(data_lines.strip().split('\n')):
# # 将字符串数据分割成行，并逐行处理
#
#
#
# # 获取第一列
# first_col = data[:, 0]
#
# # 将每个元素减去10
# first_col = first_col - 10
#
# # 找到绝对值最小的元素及其所在行的索引
# min_value = np.min(np.abs(first_col))
# min_index = np.where(np.abs(first_col) == min_value)[0][0]
#
# print("第一列数据中绝对值最小的数为：", min_value)
# print("该数所在的行索引为：", min_index)


# for i, line in enumerate(data_lines):
#     # 将行数据分割成各列数据，并获取第一列数据
#     cols = line.split()
#     first_col = float(cols[0])
#     # print(first_col)
#     # 对第一列数据进行遍历处理
#     val = abs(first_col - 10)
#     if val < min_value:
#         min_value = val
#         min_index = i
#         # print(min_value)
#         print(min_index)



