# -*- coding: utf-8 -*-
"""
@Time : 2022/11/1
@Author : 王向东
@Email : wxd1872@163.com
@File : SVR_5.py
@Project : SVR4.py
"""

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
import joblib

# df = pd.read_csv(r"D:\桌面\主动学习\22.csv")
# df = pd.read_csv(r"D:\桌面\Active Learning\data31.csv")
# df = pd.read_csv(r"D:\桌面\multiobjective\After operation\mul_feature_screening.csv")
# df = pd.read_csv(r"D:\桌面\multiobjective\After operation\mul_feature2.csv")
# df2 = pd.read_csv(r"D:\桌面\Active Learning\virtuals_screening.csv")
pd.set_option('display.max_rows', 200)
df = pd.read_csv(r"E:\pythonProject\DP\data.csv")
# df2 = pd.read_csv(r"D:\桌面\new 主动学习\n44.csv")
# array2 = df2.values
# X2 = array2[:, 2:]

# df.set_index('composition')
# print(df.shape)
# print(type(df))
# print(df.head())
# set_option('precision', 1)
# print(df.describe())
# set_option('precision', 2)
# print(df.corr(method='pearson'))

# 分离数据
array = df.values
X = array[:, 2:]
Y = array[:, 1]
# print(X)

from sklearn.preprocessing import StandardScaler
# X = StandardScaler().fit_transform(X)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
# X2 = scaler.transform(X2)

# 超参数优化
# def validate(**kwarg):
#     model = SVR(**kwarg)
#     model.fit(X, Y)
#     predictions = model.predict(X)
#     rmse = round(np.sqrt(mean_squared_error(Y, predictions)), 3)
#     r2 = round(r2_score(Y, predictions), 3)
#     return [rmse, r2]
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# trials = Trials()
# space4svm = {
#     'C': hp.uniform('C', 0, 5),
#     'gamma': hp.uniform('gamma', 0, 5),
#     'epsilon': hp.uniform('epsilon', 0, 20)
# }
# def f(params):
#     rmse = validate(**params)
#     return {'loss': rmse[0], 'status': STATUS_OK}
#
# best = fmin(fn=f, space=space4svm, algo=tpe.suggest, max_evals=1000, trials=trials)
# print(best)


# best = {'C': 3.208338778216504, 'epsilon': 0.0009261764185880446, 'gamma': 1.067072907414364}
# best = {'C': 3.8100100000912263, 'epsilon': 0.0012501369296860796, 'gamma': 0.1784263955861275}
# {'C': 4.334409351594383, 'epsilon': 0.0016220255154980705, 'gamma': 1.3880135846625161}
# best = {'C': 1.0253653855250475, 'epsilon': 0.0012460196631678862, 'gamma': 1.3761840858587355}
# best = {'C': 4.00948641635499, 'epsilon': 0.001351117326603426, 'gamma': 3.4050424087559907}
best = {'C': 4.799583906390173, 'epsilon': 0.04826507293946403, 'gamma': 0.18692917673932116}
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
    clf = SVR(**best)
    # joblib.dump(clf, 'SVR_regressor1.pkl')
    # clf = SVR()
    # clf = GaussianProcessRegressor()
    clf.fit(train_x, train_y)
    # print("("*50)
    # predictions = clf.predict(test_x)
    # print(predictions)
    # print(")" * 50)
    prediction.extend(clf.predict(test_x).tolist())
    # print(test_x)
    actual.extend(test_y.tolist())


RMSE_cv = np.sqrt(np.mean(np.square(np.array(prediction) - np.array(actual))))
R_cv = np.corrcoef(np.array(prediction), np.array(actual))[0, 1]
Rsquare_cv = r2_score(np.array(actual), np.array(prediction))
y_pred = pd.DataFrame(prediction, columns=["prediction"])
y_actual = pd.DataFrame(actual, columns=["actual"])
LOOCV_result = pd.concat([y_actual, y_pred], axis=1, sort=False)
# print(result.head())
LOOCV_result.to_csv("SVR_LOOCV.csv", index=None)


# print(np.array(prediction))
# print(np.array(actual))

# x1 = pd.DataFrame(prediction, columns=["prediction"])
# x2 = pd.DataFrame(actual, columns=["actual"])
# check_data = pd.concat([x1, x2], axis=1, sort=False)
# check_data.to_csv("check_data.csv", index=None)

print("R:%s" % R_cv)
print("R2:%s" % Rsquare_cv)
print("RMSE:%s" % RMSE_cv)
# print("得分:", Rsquare_cv)




# # 将外推空间与预测值合并
# prediction = clf.predict(X2)
# c = pd.DataFrame(prediction, columns=["zt"])
# result = pd.concat([df2, c], axis=1, sort=False)
# print(result.head())
# result.to_csv("SVR_vitual.csv", index=None)

# result.to_csv("SVR_vitual00.csv", index=None)




# model = SVR(**best).fit(X, Y)
# y_pred = model.predict(X_test)
# rmse = round(np.sqrt(mean_squared_error(Y_test, y_pred)), 3)
# r2 = round(r2_score(Y_test, y_pred), 3)
# print(rmse)
# print(r2)
# print(validate(**best))
# np.random.seed(0)
# x = np.random.randn(80, 2)
# y = x[:, 0] + 2 * x[:, 1] + np.random.randn(80)
# clf = SVR(kernel='linear', C=1.25)
# x_tran, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
# clf.fit(x_tran, y_train)
# y_hat = clf.predict(x_test)
# print("得分:", r2_score(y_test, y_hat))
# r = len(x_test) + 1
# print(y_test)
# plt.plot(np.arange(1, r), y_hat, 'go-', label="predict")
# plt.plot(np.arange(1, r), y_test, 'co-', label="real")
# plt.legend()
# plt.show()