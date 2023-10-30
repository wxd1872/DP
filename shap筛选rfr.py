# -*- coding: utf-8 -*-
"""
@Time : 2023/4/24
@Author : 王向东
@Email : wxd1872@163.com
@File : shap筛选rfr.py
@Project : feature_select.py
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
import shap
import collections, time, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl, warnings, seaborn as sns
# import xgb
# from shap import plots
from shap import plots
plots.bar

# df = pd.read_csv(r"D:\桌面\DP2\f5_shap.csv")
# df = pd.read_csv(r"D:\桌面\DP\DP2\test5.csv")
# df = pd.read_csv(r"D:\桌面\DP\f13.csv")
# df = pd.read_csv(r"D:\桌面\X1.csv")
df = pd.read_csv(r"D:\桌面\DP\feature2.csv")
print(type(df))
# feature_names = df[2:]
feature_names = df.columns[2:]
print(feature_names)
array = df.values
X = array[:, 2:]
Y = array[:, 1]
X = pd.DataFrame(X)
X = X.fillna('0')
print(X.shape)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# warnings.filterwarnings("ignore")
# sns.set()
plt.rcParams["figure.dpi"] = 600
# import shap

# from xgboost import XGBClassifier
# import data
# from sklearn.datasets import load_breast_cancer
# CLF_DATA = collections.namedtuple("DATA", "X Y")(*load_breast_cancer(return_X_y=True))
# feature_names = load_breast_cancer().feature_names
# CLF_ALGO = XGBClassifier
# model = CLF_ALGO().fit(CLF_DATA.X, CLF_DATA.Y)

# 优化后的最优参数
params = {'max_depth': 27, 'max_features': 'sqrt', 'n_estimators': 148}
RF = RandomForestRegressor(random_state=0, **params)
model = RF.fit(X, Y)
explainer = shap.TreeExplainer(model)

# shap_values = explainer.shap_values(df.iloc[:, 2:])     #获取训练集data各个样本各个特征的SHAP值。
# print(shap_values.shape)
# explainer = shap.TreeExplainer(df.iloc[:, 2:])
shap_values = explainer.shap_values(df.iloc[:, 2:])
# shap_values2 = explainer(df.iloc[:, 2:])
print(shap_values)
# shap.summary_plot(shap_values, df.iloc[:, 2:], feature_names=feature_names, plot_type="bar")
shap.summary_plot(shap_values, df.iloc[:, 2:], feature_names=feature_names, plot_type="bar", max_display=25)
fig = plt.gcf() # 获取后面图像的句柄
shap.summary_plot(shap_values, df.iloc[:, 2:], plot_type="violin")




#
# shap.matplotlib.pyplot.savefig("shap1d_bar.png",dpi=800, bbox_inches = 'tight')
#
# plt.figure(dpi=1200,figsize=(1,2)) # 设置图片的清晰度
# plt.xlim(-0.6, 0.6)
# fig = plt.gcf() # 获取后面图像的句柄
# shap.summary_plot(shap_values, df.iloc[:, 2:])


# shap.summary_plot(shap_values, df.iloc[:, 2:], feature_names=feature_names, plot_type="layered_violin", color='coolwarm', max_display=15, show=False)
# # plt.savefig("name.png",dpi=800, bbox_inches = 'tight')
# shap.matplotlib.pyplot.savefig("shap1d.png", dpi=800, bbox_inches = 'tight')


# plt.figure(dpi=1200) # 设置图片的清晰度
# fig = plt.gcf() # 获取后面图像的句柄
# shap.summary_plot(shap_values, df.iloc[:, 2:], plot_type="bar")

# plt.figure(dpi=1200) # 设置图片的清晰度


#
# explainer = shap.explainers.Tree(model, X, feature_perturbation="interventional", feature_names=feature_names)
# shap_values = explainer(X)
# shap.plots.bar(shap_values)
# shap.plots.waterfall(shap_values[0, :])
# shap.plots.violin(shap_values.values, X)
# shap.plots.beeswarm(shap_values)
# shap.plots.heatmap(shap_values)
# shap.plots.decision(shap_values.base_values[0], shap_values.values)
# shap.plots.partial_dependence(1, model.predict, X, feature_names=feature_names)



# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)
#
# explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X)
#
# # s = Shap().fit(clf, X, Y, feature_names=columns.values.tolist())
# s.bar(max_display=10)
# # shap_values = s.shap_values.values
#
# shap_values = pd.DataFrame(shap_values, columns=columns, index=dataset.index).iloc[:, s.feature_index[:9]]
#
# shap_values.to_excel("shap_values.xlsx")