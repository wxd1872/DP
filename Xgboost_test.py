# -*- coding: utf-8 -*-
"""
@Time : 2023/3/15
@Author : 王向东
@Email : wxd1872@163.com
@File : Xgboost_test.py
@Project : DP
"""
import pandas
from sklearn import preprocessing
import pandas as pd
import numpy as np
from xgboost import XGBRegressor as XGBR
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

# 分离数据
array = df.values
X = array[:, 2:]
Y = array[:, 1]
# print(X)

# 标准化
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)
random_state= 0
params = {'learning_rate': 0.1, 'n_estimators': 565}
# params = {'max_depth': 27, 'max_features': 'sqrt', 'n_estimators': 148}

# 对独立测试集进行验证
R2_train = []
R2_test = []
r_test = []
for i in range(100):
    validation_size = 0.2
    # seed = 8
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_size)
    # clf = SVR(C=4.7995, epsilon=0.0482650729, gamma=0.18692917673)
    # clf = SVR(C=3.208338778216504, epsilon=0.0009261764185880446, gamma=1.067072907414364)
    # clf = KNeighborsRegressor()
    clf = XGBR(seed = random_state, **params)
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

    r_test.append(R_test)
    R2_train.append(Rsquare_train)
    R2_test.append(Rsquare_test)
    # sio.savemat('train.mat',{'y_train':y_train})

print(len(r_test))
xx1_train = np.array(R2_train).mean()
xx1_test = np.array(R2_test).mean()
r_mean = np.array(r_test).mean()

print("*"*100)
print("R2_train:%s" % xx1_train)
print("R2_test:%s" % xx1_test)
print("r_test:%s" % r_mean)