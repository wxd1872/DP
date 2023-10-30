# -*- coding: utf-8 -*-
"""
@Time : 2022/11/17
@Author : 王向东
@Email : wxd1872@163.com
@File : feature_select.py
@Project : SVR4.py
"""

# 特征初步筛选
import joblib
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
from sklearn.feature_selection import VarianceThreshold, RFE
# from sklearn.feature_selection import RFE
import seaborn as sns


pd.set_option('display.expand_frame_repr', False)
# df = pd.read_csv(r"D:\桌面\Active Learning\data1.csv")
# df = pd.read_csv(r"D:\桌面\multiobjective\After operation\multi_features.csv")
# df = pd.read_csv(r"D:\桌面\DP\312.csv")
# df = pd.read_csv(r"D:\桌面\DP2\f1.csv")
df = pd.read_csv(r"D:\桌面\DP\f11.csv")
# df2 = pd.read_csv(r"D:\桌面\new 主动学习\n44.csv")
# array2 = df2.values
# X2 = array2[:, 3:]
# array = df.values
print(df.shape)
X = df.iloc[:, 2:]
# 保留特征名称
all_name = X.columns.values.tolist()  # 获得所有的特征名称
# X = pd.DataFrame(X)


print(X.shape)
Y = df.values[:, 1]
print(Y)


# 删除方差为0的列
# print(type(X))
selector = VarianceThreshold()
# selector = VarianceThreshold(threshold = (0.5))
X = selector.fit_transform(X)

# 输出筛选后特征的索引值
select_name_index = selector.get_support(indices=True)  # 留下特征的索引值，list格式
print("@"*100)
print(select_name_index)
select_name = []
for i in select_name_index:
    select_name.append(all_name[i])
# print(select_name)
X = pd.DataFrame(X)
X.columns = select_name
print(X.head())
print(X.shape)

df = X.apply(lambda x: x.astype(float))
df.to_csv('mul_VarianceThreshold.csv', index=True)


# df = pd.read_csv(r"D:\桌面\Active Learning\VarianceThreshold2.csv")
print("*"*60)
# print(df.head())
# print(df.shape)
# Create correlation matrix
# corr_matrix1 = df.corr()
# print(corr_matrix1)
corr_matrix = df.corr().abs()  #计算相关系数并取绝对值
# print(corr_matrix)
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
print(upper)
# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
# print(type[to_drop])
# Drop features
# df = df.drop(['n_Seebeck(180)'], axis=1, inplace=True)
df = df.drop(to_drop, axis=1)
print(df.shape)
print(df)
df.to_csv('feature_screening1.csv', index=True)







# print(df1.info())
# 创建相关度矩阵
# print(df.corr())
# corr_matrix = df1.corr()
# corr_matrix = df1.corr().abs()
# # print(corr_matrix)
# # 选择相关度矩阵的上三角
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
# print(upper)


#
# # 寻找相关度大于 0.95 的特征列的索引
# to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
#
# # 丢弃特征
# df.drop(df.columns[to_drop], axis=1)
# print(df)



# 方法2.计算特征间的pearson相关系数，画heatmap图
# plt.figure(figsize=(25, 25))
# corr_values1 = df.corr()  # pandas直接调用corr就能计算特征之间的相关系数
# print(corr_values1)
# sns.heatmap(corr_values1, annot=True, vmax=1, square=True, cmap="Blues", fmt='.2f')
# plt.tight_layout()
# plt.savefig('prepare_data/columns37.png',dpi=600)
# plt.show()





# # Create correlation matrix
# corr_matrix = X.corr().abs()
#
# # Select upper triangle of correlation matrix
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# print(upper)
# # Find features with correlation greater than 0.95
# to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
# # print(type[to_drop])
# # Drop features
# # df = df.drop(['n_Seebeck(180)'], axis=1, inplace=True)
# df = df.drop(to_drop, axis=1)
# print(df)


# best = {'C': 3.208338778216504, 'epsilon': 0.0009261764185880446, 'gamma': 1.067072907414364}
# svr = SVR(**best)
# selector =  RFE(svr,)
# estimator = SVR(**best)
# estimator = SVR(kernel="linear")
# selectors = RFE(estimator, n_features_to_select=6, step=5)
# selectors = selectors.fit(X, Y)

#
# print(selectors.support_)
# print(selectors.ranking_)
# score = []
# for i in range(3, 51, 5):
#     svr = SVR(**best)
#     X_wrapper = RFE(svr, n_features_to_select=i, step=5).fit_transform(X, Y)
#     once = cross_val_score(svr, X_wrapper, Y, cv=5).mean()
#     score.append(once)
# plt.figure(figsize=[20, 5])
# plt.plot(range(3, 51, 5), score)
# plt.xticks(range(3, 51,5))
# plt.show()


# select_name_index = selector.get_support(indices=True)  # 留下特征的索引值，list格式
# select_name = []
# for i in select_name_index:
#     select_name.append(all_name[i])
# # print(select_name)
# X = pd.DataFrame(X)
# X.columns = select_name
# print(X.head())
# print(X.shape)
# scaler = StandardScaler()
# scaler.fit(X)
# Y = pd.DataFrame(Y1)
# alldata = pd.concat([X, Y], axis=1)
# Virtual = scaler.transform(X2)
# Virtual = pd.DataFrame(Virtual)
