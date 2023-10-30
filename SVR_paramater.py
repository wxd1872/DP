# -*- coding: utf-8 -*-
"""
@Time : 2022/10/7
@Author : 王向东
@Email : wxd1872@163.com
@File : SVR2.py
@Project : miaoshufu.py
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

pd.set_option('display.max_rows', 200)
# pd.set_option('display.expand_frame_repr', False)
# df = pd.read_csv(r"D:\桌面\DP\After run\3121_feature_screening.csv")
df = pd.read_csv(r"E:\pythonProject\DP\data.csv")
# df = pd.read_csv(r"D:\桌面\multiobjective\After operation\mul_feature_screening.csv")
# df = pd.read_csv(r"D:\桌面\multiobjective\After operation\mul_feature2.csv")
# df = pd.read_csv(r"D:\桌面\主动学习\22.csv")
# df = pd.read_csv(r"D:\桌面\new 主动学习\new3.csv")
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
# print(type(X))
X = pd.DataFrame(X)
# print(type(X))
# 检查缺失值和用0去填补缺失值
print(X.isnull().any())
X = X.fillna('0')
print(X.shape)
# print(X.isnull().any())

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

def validate(**kwarg):
    model = SVR(**kwarg)
    model.fit(X, Y)
    predictions = model.predict(X)
    rmse = round(np.sqrt(mean_squared_error(Y, predictions)), 3)
    r2 = round(r2_score(Y, predictions), 3)
    return [rmse, r2]
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

trials = Trials()
space4svm = {
    'C': hp.uniform('C', 0, 5),
    'gamma': hp.uniform('gamma', 0, 5),
    'epsilon': hp.uniform('epsilon', 0, 20)
}
def f(params):
    rmse = validate(**params)
    return {'loss': rmse[0], 'status': STATUS_OK}

best = fmin(fn=f, space=space4svm, algo=tpe.suggest, max_evals=100, trials=trials)
model = SVR(**best).fit(X, Y)
# y_pred = model.predict(X_test)
# rmse = round(np.sqrt(mean_squared_error(Y_test, y_pred)), 3)
# r2 = round(r2_score(Y_test, y_pred), 3)
# print(rmse)
# print(r2)
print(best)
print(validate(**best))



# np.random.seed(0)
#
# x = np.random.randn(80, 2)
# y = x[:, 0] + 2 * x[:, 1] + np.random.randn(80)
#
# clf = SVR(kernel='linear', C=1.25)
# x_tran, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
# clf.fit(x_tran, y_train)
# y_hat = clf.predict(x_test)
#
# print("得分:", r2_score(y_test, y_hat))
#
# r = len(x_test) + 1
# print(y_test)
# plt.plot(np.arange(1, r), y_hat, 'go-', label="predict")
# plt.plot(np.arange(1, r), y_test, 'co-', label="real")
# plt.legend()
# plt.show()