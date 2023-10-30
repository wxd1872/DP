# -*- coding: utf-8 -*-
"""
@Time : 2023/3/7
@Author : 王向东
@Email : wxd1872@163.com
@File : POSCAR_CIF.py
@Project : GA_FS_HP.py
"""
import io
import os
import numpy as np
import pandas as pd
from ase.io import read, write
from ase import io
from pymatgen.core import Structure
# import os
import shutil

# 创建一个名为 my_folder 的文件夹
# os.mkdir('my_folder')
# 导入poscar文件
rootdir = 'E:\py\wumagPOSCAR'
filepath = os.listdir(rootdir)
# 列出文件夹下所有的目录与文件
composition = []
# for i in range(0, 5):
for i in range(0, len(filepath)):
    file = os.path.join(rootdir, filepath[i])
    print(file)
    # print(filepath[i])
    jobname = filepath[i]
    s = Structure.from_file(file)
    # s.to(filename = "hf1.cif")
    s.to(filename = jobname+'.cif')
    # shutil.copy(jobname+'.cif', 'my_folder/')

# 打印 my_folder 文件夹中的文件列表
# print(os.listdir('my_folder'))
    # s.to(filename= "hf1.cif")
    # s.to("E:\py\test.cif", fmt="cif")






# 将需要导出的文件复制到 my_folder 文件夹中
# shutil.copy('file1.txt', 'my_folder/')
# shutil.copy('file2.txt', 'my_folder/')
# 继续复制其他文件

# 打印 my_folder 文件夹中的文件列表
# print(os.listdir('my_folder'))


    # io.write('file', io.read("AB.cif"), format='vasp', direct=True)



#     print(file)
#     with open(file, 'r') as fpos:
#         print(fpos)
#         pos = fpos.readlines()


# atoms = io.read(jobname+'.cif')
# atoms.write('POSCAR', format = 'vasp')

# import os

# 设置POSCAR文件目录和CIF文件目录
# poscar_dir ='E:\py\POSCAR_t'
# cif_dir = 'E:\py\cif'
#
# # 遍历POSCAR文件目录，批量转换POSCAR文件为CIF文件
# for file in os.listdir(poscar_dir):
#     if file.endswith(".POSCAR"):
#         # 构建输入文件路径和输出文件路径
#         poscar_path = os.path.join(poscar_dir, file)
#         cif_path = os.path.join(cif_dir, file.replace(".POSCAR", ".cif"))
#         # 执行转换命令
#         os.system("/path/to/vasp/scripts/pos2cif.pl {} > {}".format(poscar_path, cif_path))
