# -*- coding: utf-8 -*-
"""
@Time : 2023/4/20
@Author : 王向东
@Email : wxd1872@163.com
@File : sub_number.py
@Project : feature_select.py
"""



import re

formula = "Cs4H4O8Si2"

digits = re.findall(r'\d+\.?\d*', formula)
non_digits = re.findall(r'[A-Za-z]+', formula)

print(digits)
print(non_digits)

digits = [4, 4, 8, 2]
non_digits = ['Cs', 'H', 'O', 'Si']
formula = ''.join([non_digits[i] + str(digits[i]) for i in range(len(digits))])
print(formula)

