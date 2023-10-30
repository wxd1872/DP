# -*- coding: utf-8 -*-
"""
@Time : 2022/11/8
@Author : 王向东
@Email : wxd1872@163.com
@File : Xgb2.py
@Project : SVR4.py
"""
import xgboost as xgb
from xgboost import XGBRegressor as XGBR
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
# df = pd.read_csv(r"D:\桌面\new 主动学习\new4.csv")
# df2 = pd.read_csv(r"D:\桌面\new 主动学习\n3.csv")
# df = pd.read_csv(r"D:\桌面\Active Learning\data1.csv")

# df = pd.read_csv(r"D:\桌面\Active Learning\data24.csv")
# # print(df.shape[0])
# df2 = pd.read_csv(r"D:\桌面\Active Learning\virtuals_screening.csv")
pd.set_option('display.max_rows', 200)
df = pd.read_csv(r"E:\pythonProject\DP\data.csv")
# df = pd.read_csv(r"D:\桌面\multiobjective\After operation\mul_feature2.csv")
# df = pd.read_csv(r"D:\桌面\multiobjective\After operation\mul_feature_screening.csv")
# df = pd.read_csv(r"D:\桌面\Active Learning\data31.csv")
# df2 = pd.read_csv(r"D:\桌面\Active Learning\virtuals_screening11.csv")
# array2 = df2.values
# X2 = array2[:, 2:]
# 分离数据
array = df.values
X = array[:, 2:]
Y = array[:, 1]
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
# X2 = scaler.transform(X2)


scoring = 'neg_mean_squared_error'
param_grid = {"n_estimators": [*range(300, 650, 5)], 'learning_rate': [*np.arange(0.1, 0.2, 0.01)]}
# param_grid ={'learning_rate': [0.1], 'n_estimators': [186]}
# param_grid = {"n_estimators": [120], 'learning_rate': [0.21]}
model = XGBR(random_state=0)
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold, scoring = scoring, n_jobs=4)
# grid.fit(X_train, Y_train)
grid.fit(X, Y)

print(grid.best_params_)
print(grid.best_score_)
params = grid.best_params_

# 纯交叉验证
random_state=0
# params = {'learning_rate': 0.18000000000000005, 'n_estimators': 195}

R2_train = []
R2_test = []
r_test = []
kf = KFold(100, shuffle=True)
# kf = KFold(df.shape[0], shuffle=True)
prediction = []
actual = []
for train_index, test_index in kf.split(X, Y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    train_x, train_y = X[train_index], Y[train_index]
    test_x, test_y = X[test_index], Y[test_index]
    clf = XGBR(seed = random_state, **params)
    # clf = XGBR(**params)
    clf.fit(train_x, train_y)
    prediction.extend(clf.predict(test_x).tolist())
    actual.extend(test_y.tolist())
RMSE_cv = np.sqrt(np.mean(np.square(np.array(prediction) - np.array(actual))))
R_cv = np.corrcoef(np.array(prediction), np.array(actual))[0, 1]
Rsquare_cv = r2_score(np.array(actual), np.array(prediction))

print("R:%s" % R_cv)
print("R2:%s" % Rsquare_cv)
print("RMSE:%s" % RMSE_cv)

y_pred = pd.DataFrame(prediction, columns=["prediction"])
y_actual = pd.DataFrame(actual, columns=["actual"])
LOOCV_result = pd.concat([y_actual, y_pred], axis=1, sort=False)
# print(result.head())
LOOCV_result.to_csv("XGB_LOOCV.csv", index=None)

# prediction = clf.predict(X2)
# c = pd.DataFrame(prediction, columns=["zt"])
# result = pd.concat([df2, c], axis=1, sort=False)
# print(result.head())
# result.to_csv("xgb2_vitual1.csv", index=None)
#
# joblib.dump(clf, 'Xgb2_regressor.pkl')