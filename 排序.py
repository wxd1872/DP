# -*- coding: utf-8 -*-
"""
@Time : 2023/6/10
@Author : 王向东
@Email : wxd1872@163.com
@File : 排序.py
@Project : Deal_dos_feature.py
"""


# 对随机森林进行参数优化和交叉验证
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import re
from itertools import groupby
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import os.path as path
import argparse
import sys
from pandas import set_option
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import check_random_state
from sklearn.pipeline import make_pipeline
import joblib
from fml.feature_selection import Shap



df1 = pd.read_excel('D:\桌面\dtest\qtable1.xlsx', index_col=0)
df2 = pd.read_excel('D:\桌面\dtest\qtable2.xlsx', index_col=0)
# print(df1)

table = []
for index, row in df1.iterrows():
    row2 = df2.loc[index]
    # print(row, row2)
    new_series = pd.concat([row, row2], axis=0)
    df = new_series.to_frame()
    # print(new_series)
    # 使用T属性进行转置
    transposed = df.T
    print(transposed)
    # concatenated_df = concatenated_df.reset_index()
    # print(type(new_series))
    # print("1" * 100)
    table.append(transposed)
result = pd.concat(table)
print("3"*100)
print(result)

result.to_excel("result.xlsx")






# print(type(row))
# print("2"*100)
# print(type(row2))





# df1_values = df1.values.tolist()
# df2_values = df2.values.tolist()
# print(df2_values)
#

# for row1, row2 in zip(df1_values, df2_values):
#     print(row1, row2)

# print(type(df1))




# merged_df = pd.concat([df1, df2], axis=0, ignore_index=True)
# merged_df.sort_values(by='composition', inplace=True)
# print(merged_df)
# merged_df.to_excel('merged_tables.xlsx')