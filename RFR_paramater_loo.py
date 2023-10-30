# -*- coding: utf-8 -*-
"""
@Time : 2022/11/2
@Author : 王向东
@Email : wxd1872@163.com
@File : RFR.py
@Project : SVR4.py
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


# df = pd.read_csv(r"D:\桌面\主动学习\22.csv")
# df = pd.read_csv(r"D:\桌面\new 主动学习\new33.csv")
# df2 = pd.read_csv(r"D:\桌面\new 主动学习\n44.csv")
# array2 = df2.values
# X2 = array2[:, 3:]
# df = pd.read_csv(r"D:\桌面\multiobjective\After operation\mul_feature_screening.csv")
# df = pd.read_csv(r"E:\pythonProject\DP\data.csv")
# df = pd.read_csv(r"D:\桌面\DP2\f4.csv")
# df = pd.read_csv(r"D:\桌面\DP2\f5_shap.csv")

# df = pd.read_csv(r"D:\桌面\DP\f13.csv")
df = pd.read_csv(r"D:\桌面\DP\feature2.csv")
columns = df.columns[1:]

# df = pd.read_csv(r"D:\桌面\Active Learning\data31.csv")
# df2 = pd.read_csv(r"D:\桌面\Active Learning\virtuals_screening11.csv")
# df = pd.read_csv(r"D:\桌面\Active Learning\data20.csv")

# 分离数据
array = df.values
X = array[:, 2:]
Y = array[:, 1]

# print(X)
# df2 = pd.read_csv(r"D:\桌面\Active Learning\virtuals_screening.csv")
# array2 = df2.values
# X2 = array2[:, 2:]
X = pd.DataFrame(X)
X = X.fillna('0')
print(X.shape)
X = StandardScaler().fit_transform(X)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
# X2 = scaler.transform(X2)

#将数据集分割为训练集和测试集
# Y = data.quality
# X = data.drop('quality', axis=1)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123,)
# X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=123, stratify=Y )

# 创建管道
# pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=184))
# pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor())
model = RandomForestRegressor(random_state=0)

# 声明模型需要关注的超参
# {'max_depth': 6, 'max_features': 'sqrt', 'n_estimators': 142}

parameters = {'max_features': ['sqrt']
                    , 'max_depth': [40, 30, 29, 26, 25, 20, 24, 23, 22, 21, 100, 28, 27]
                    , "n_estimators": [*range(140, 150, 1)]}

# parameters = {'max_depth': 26, 'max_features': 'sqrt', 'n_estimators': 147}
# 用K折交叉检验和网格搜索对模型进行训练和调参
grid = GridSearchCV(estimator=model, param_grid=parameters, cv=5, scoring = 'neg_mean_squared_error', n_jobs=3)
grid.fit(X, Y)
print(grid.best_params_)
print(grid.best_score_)
params = grid.best_params_

# 优化后的最优参数
# params = {'max_depth': 27, 'max_features': 'sqrt', 'n_estimators': 148}
# print(clf.best_score_)
# neg_mean_squared_error
# print(clf. best_params_)
# print(estimator.get_params().keys())
# clf.fit(X, Y)
# # 利用得到的模型进行预测并用预测结果对模型进行性能评估
# pred = clf.predict(X_test)
# r2_score(Y_test, pred)
# mean_squared_error(Y_test, pred)
#
# print(r2_score(Y_test, pred))
# print(Y_test)
# print(pred)
# 保存模型以便将来使用
# joblib.dump(clf, 'rf_regressor.pkl')
# 取模型来使用
# clf2 = joblib.load('rf_regressor.pkl')
# for i in range(1, 201, 10):
#     print(i)
kf = KFold(100, shuffle=True)
# kf = KFold(df.shape[0], shuffle=True)
prediction = []
actual = []
for train_index, test_index in kf.split(X, Y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    train_x, train_y = X[train_index], Y[train_index]
    test_x, test_y = X[test_index], Y[test_index]
    # print("$"*50)
    # print(type(test_x.tolist()))
    # clf = KNeighborsRegressor()
    # clf = SVR(**best)
    # clf = SVR()
    # clf = GaussianProcessRegressor()
    clf = RandomForestRegressor(random_state=0, **params)
    clf.fit(train_x, train_y)
    # print("("*50)
    # predictions = clf.predict(test_x)
    # print(predictions)
    # print(")" * 50)
    prediction.extend(clf.predict(test_x).tolist())
    # print(test_x)
    actual.extend(test_y.tolist())
    # print("&" * 100)
    # print(len(prediction))
    # print(prediction)
    # print("!"*100)
    # print(actual)
    # print(len(actual))
    # Rsquare_cv = r2_score(np.array(actual), np.array(prediction))
    # print(Rsquare_cv)
RMSE_cv = np.sqrt(np.mean(np.square(np.array(prediction) - np.array(actual))))
R_cv = np.corrcoef(np.array(prediction), np.array(actual))[0, 1]
Rsquare_cv = r2_score(np.array(actual), np.array(prediction))
y_pred = pd.DataFrame(prediction, columns=["prediction"])
y_actual = pd.DataFrame(actual, columns=["actual"])
LOOCV_result = pd.concat([y_actual, y_pred], axis=1, sort=False)
# print(result.head())
LOOCV_result.to_csv("RFR_LOOCV.csv", index=None)
print("R:%s" % R_cv)
print("R2:%s" % Rsquare_cv)
print("RMSE:%s" % RMSE_cv)






# print("得分:", Rsquare_cv)
# print(clf.best_params_)
# y_pred = pd.DataFrame(prediction, columns=["prediction"])
# y_actual = pd.DataFrame(actual, columns=["actual"])
# LOOCV_result = pd.concat([y_actual, y_pred], axis=1, sort=False)
# # print(result.head())
# LOOCV_result.to_csv("RFR_LOOCV.csv", index=None)
# #
# prediction = clf.predict(X2)
# c = pd.DataFrame(prediction, columns=["zt"])
# result = pd.concat([df2, c], axis=1, sort=False)
# print(result.head())
# result.to_csv("RER_vitual.csv", index=None)

#
# joblib.dump(clf, 'rfr_regressor.pkl')