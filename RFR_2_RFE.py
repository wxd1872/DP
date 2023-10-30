# -*- coding: utf-8 -*-
"""
@Time : 2022/11/23
@Author : 王向东
@Email : wxd1872@163.com
@File : RFR_2_RFE.py
@Project : SVR4.py
"""
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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# 将数据铺开
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 200)

# df = pd.read_csv(r"D:\桌面\Active Learning\data1.csv")
# df = pd.read_csv(r"D:\桌面\multiobjective\After operation\mul_feature_screening_tough.csv")
# df = pd.read_csv(r"D:\桌面\DP2\DP\After run\312_feature_screening.csv")
# df = pd.read_csv(r"D:\桌面\DP\f12.csv")
df = pd.read_csv(r"D:\桌面\X1.csv")

X = df.iloc[:, 2:]
# 保留特征名称
# all_name = X.columns.values.tolist()  # 获得所有的特征名称
# 转换X的数据类型
print(X.shape)
Y = df.iloc[:, 1]


# Y = df.values[:, 1]
X = pd.DataFrame(X)
# print(type(X))
# 判断特征中是否有缺失值
# print(X.isnull().any())
# 对特征中的缺失值用0填充
X = X.fillna('0')
# print(Y)
RFR_ = RandomForestRegressor(max_depth=23, max_features='sqrt', n_estimators=146, random_state=0)
RFR_.fit(X, Y)

# 特征重要性排序
features = X.columns
# print(features)
feature_importances = RFR_.feature_importances_
features_df = pd.DataFrame({'Features': features, 'Importance': feature_importances})
features_df.sort_values('Importance', inplace=True, ascending=False)

# 输出特征重要性排序
# print(feature_importances)
print(features_df)
print(features_df.shape)
# print(features_df.head())
features_df.to_csv('features_df.csv', index=True)
print("2"*100)

# 将筛选的特征选出
# Features = features_df["Features"]
Features = features_df.values[:, 0]
# select_name = Features.iloc[:, 0]
print(Features)
print(type(Features))
# select_name = Features

# select_name = Features[:21:1]
select_name = Features[:10:1]

# # print(Features[:52:1])
X_select = X[select_name]
print(X[select_name])
print(X_select)
X_select.to_csv('X1.csv', index=True)
# print(type(X_select))
# print("1"*100)
# print(X_select.shape)





# select_name = []
# # for i in Features:
# #     select_name.append(Features[i])
# # print(select_name)
# data = pd.concat([Y, X_select], axis=1)
# data.to_csv('ff1.csv', index=True)

# data = pd.concat([Y, X_select], axis=1)
# y = data.iloc[:, 0]
# x = data.iloc[:, 1:]
# x = pd.DataFrame(x)
# print(type(x))
#
# array = data.values
# x = array[:, 1:]
# y = array[:, 0]
# print(type(x))

# X = StandardScaler().fit_transform(X)
# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)

# # 验证筛选出的特征
# score = []
# R2 = []
# kf = KFold(100, shuffle=True)
# prediction = []
# actual = []
# for train_index, test_index in kf.split(x, y):
#     # print("TRAIN:", train_index, "TEST:", test_index)
#     train_x, train_y = x[train_index], y[train_index]
#     test_x, test_y = x[test_index], y[test_index]
#         # print("$"*50)
#         # print(type(test_x.tolist()))
#         # clf = KNeighborsRegressor()
#     clf = RFR_
#         # clf = SVR()
#         # clf = GaussianProcessRegressor()
#     clf.fit(train_x, train_y)
#         # print("("*50)
#         # predictions = clf.predict(test_x)
#         # print(predictions)
#         # print(")" * 50)
#     prediction.extend(clf.predict(test_x).tolist())
#         # print(test_x)
#     actual.extend(test_y.tolist())
#         # print("&" * 100)
#         # print(len(prediction))
#         # print(prediction)
#         # print("!"*100)
#         # print(actual)
#         # print(len(actual))
#         # Rsquare_cv = r2_score(np.array(actual), np.array(prediction))
#         # print(Rsquare_cv)
# RMSE_cv = np.sqrt(np.mean(np.square(np.array(prediction) - np.array(actual))))
# R_cv = np.corrcoef(np.array(prediction), np.array(actual))[0, 1]
# Rsquare_cv = r2_score(np.array(actual), np.array(prediction))
# y_pred = pd.DataFrame(prediction, columns=["prediction"])
# y_actual = pd.DataFrame(actual, columns=["actual"])
# LOOCV_result = pd.concat([y_actual, y_pred], axis=1, sort=False)
#     # print("R:%s" % R_cv)
# print("R2:%s" % Rsquare_cv)
#     # print("RMSE:%s" % RMSE_cv)
# score.append(RMSE_cv)
# R2.append(Rsquare_cv)
# rmse = pd.DataFrame(score, columns=["rmse"])
# r2 = pd.DataFrame(R2, columns=["r2"])
# rmse.to_csv('rmse.csv', index=True)
# r2.to_csv('r2.csv', index=True)

# # 对包装法画学习曲线
# score = []
# for i in range(1, 59, 5):
#     X_wrapper = RFE(RFR_, n_features_to_select=i, step=5).fit_transform(X, Y)
#     once = cross_val_score(RFR_, X_wrapper, Y, cv=5).mean()
#     score.append(once)
# plt.figure(figsize=[20, 5])
# plt.plot(range(1, 59, 5), score)
# # plt.xticks(range(1, 113, 3))
# plt.show()











# selector = RFE(RFR_, n_features_to_select=51, step=1).fit(X, Y)
# print(selector.support_.sum())
# print(selector.ranking_)
# X_wrapper = selector.transform(X)
# cross_val_score(RFR_, X_wrapper, Y, cv=5).mean()

# score = []
# R2 = []
# for i in range(1, 29, 1):
#     # print("%"*100)
#     print(i)
#     X_wrapper = RFE(RFR_, n_features_to_select=i, step=1).fit_transform(x, y)
#     # print(X_wrapper)
#     # print(X_wrapper.ranking_)
#     # once = cross_val_score(RFR_, X_wrapper, y, scoring='r2', cv=5).mean()
#     kf = KFold(100, shuffle=True)
#     prediction = []
#     actual = []
#     for train_index, test_index in kf.split(X_wrapper, y):
#         # print("TRAIN:", train_index, "TEST:", test_index)
#         train_x, train_y = X_wrapper[train_index], Y[train_index]
#         test_x, test_y = X_wrapper[test_index], Y[test_index]
#         # print("$"*50)
#         # print(type(test_x.tolist()))
#         # clf = KNeighborsRegressor()
#         clf = RFR_
#         # clf = SVR()
#         # clf = GaussianProcessRegressor()
#         clf.fit(train_x, train_y)
#         # print("("*50)
#         # predictions = clf.predict(test_x)
#         # print(predictions)
#         # print(")" * 50)
#         prediction.extend(clf.predict(test_x).tolist())
#         # print(test_x)
#         actual.extend(test_y.tolist())
#         # print("&" * 100)
#         # print(len(prediction))
#         # print(prediction)
#         # print("!"*100)
#         # print(actual)
#         # print(len(actual))
#         # Rsquare_cv = r2_score(np.array(actual), np.array(prediction))
#         # print(Rsquare_cv)
#     RMSE_cv = np.sqrt(np.mean(np.square(np.array(prediction) - np.array(actual))))
#     R_cv = np.corrcoef(np.array(prediction), np.array(actual))[0, 1]
#     Rsquare_cv = r2_score(np.array(actual), np.array(prediction))
#     y_pred = pd.DataFrame(prediction, columns=["prediction"])
#     y_actual = pd.DataFrame(actual, columns=["actual"])
#     LOOCV_result = pd.concat([y_actual, y_pred], axis=1, sort=False)
#     # print("R:%s" % R_cv)
#     print("R2:%s" % Rsquare_cv)
#     # print("RMSE:%s" % RMSE_cv)
#     score.append(RMSE_cv)
#     R2.append(Rsquare_cv)
# rmse = pd.DataFrame(score, columns=["rmse"])
# r2 = pd.DataFrame(R2, columns=["r2"])
# rmse.to_csv('rmse.csv', index=True)
# r2.to_csv('r2.csv', index=True)



# print(score)
# plt.figure(figsize=[20, 5])
# plt.plot(range(2, 90, 1), score)
# plt.xticks(range(2, 90, 1))
# plt.show()