# -*- coding: utf-8 -*-
"""
@Time : 2022/8/24
@Author : 王向东
@Email : wxd1872@163.com
@File : SVR.py
@Project : cluster.py
"""

import pandas
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

# df = pd.read_csv(r"D:\桌面\new 主动学习\new3.csv")

pd.set_option('display.max_rows', 200)
df = pd.read_csv(r"E:\pythonProject\DP\data.csv")
# df = pd.read_csv(r"D:\桌面\Active Learning\data1.csv")
# df.set_index('composition')
# print(df.shape)
# print(type(df))
# print(df.head())

# 分离数据
array = df.values
X = array[:, 2:]
Y = array[:, 1]
# print(X)

# 标准化
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

# 对独立测试集进行验证
R2_train = []
R2_test = []
r_test = []
for i in range(100):
    validation_size = 0.2
    # seed = 8
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_size)

    # best = fmin(fn=f, space=space4svm, algo=tpe.suggest, max_evals=1000, trials=trials)
    # model = SVR().fit(X_train, Y_train)
    # y_pred = model.predict(X_test)
    # rmse = round(np.sqrt(mean_squared_error(Y_test, y_pred)), 3)
    # r2 = round(r2_score(Y_test, y_pred), 3)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=10)
    # clf = SVR(C=3.3, epsilon=0.0003, gamma=4.67)
    # {'C': 4.799583906390173, 'epsilon': 0.04826507293946403, 'gamma': 0.18692917673932116}
    clf = SVR(C=4.7995, epsilon=0.0482650729, gamma=0.18692917673)
    # clf = SVR(C=3.208338778216504, epsilon=0.0009261764185880446, gamma=1.067072907414364)
    # clf = KNeighborsRegressor()
    clf.fit(X_train, Y_train)
    # train
    prediction=clf.predict(X_train)
    actual=Y_train
    RMSE_train = np.sqrt(np.mean(np.square(np.array(prediction) - np.array(actual))))
    # R_train = np.corrcoef(np.array(prediction), np.array(actual))[0, 1]
    Rsquare_train = r2_score(np.array(actual), np.array(prediction))
    # test
    prediction = clf.predict(X_test)
    # print(prediction)
    # print(type(prediction))

    prediction = (prediction.tolist())
    # print(prediction)
    # print(type(prediction))
    # modelpath=mainpath+"/"+model_class.__name__+".pkl"
    # joblib.dump(clf,modelpath)
    actual = Y_test
    # print(type(actual))
    actual = actual.tolist()
    # print(actual)
    # print(type(actual))

    RMSE_test = np.sqrt(np.mean(np.square(np.array(prediction) - np.array(actual))))
    R_test = np.corrcoef(np.array(prediction), np.array(actual))[0, 1]
    Rsquare_test = r2_score(np.array(actual), np.array(prediction))

    # print(R_train)
    # print(Rsquare_train)

    # print(R_test)
    # print(Rsquare_test)
    r_test.append(R_test)
    R2_train.append(Rsquare_train)
    R2_test.append(Rsquare_test)
    # sio.savemat('train.mat',{'y_train':y_train})
# print(list11)
# xx1_train = np.array(R2_train)
# xx1_train = xx1_train.mean()
print(len(r_test))
xx1_train = np.array(R2_train).mean()
xx1_test = np.array(R2_test).mean()
r_mean = np.array(r_test).mean()

print("*"*100)

print("R2_train:%s" % xx1_train)
print("R2_test:%s" % xx1_test)
print("r_test:%s" % r_mean)
# print(xx1_train)
# print(xx1_test)
# print(r_mean)


