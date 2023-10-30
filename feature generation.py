# -*- coding: utf-8 -*-
"""
@Time : 2023/1/27
@Author : 王向东
@Email : wxd1872@163.com
@File : feature processing1.py
@Project : SVR4.py
"""

if __name__ == '__main__':
    import pandas as pd
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers import composition as cf
    from matminer.featurizers.conversions import StrToComposition

    # df_m = pd.read_csv(r"D:\桌面\主动学习\virtuals1.csv")
    # df_m = pd.read_csv(r"D:\桌面\multiobjective\toughness.csv")
    df_m = pd.read_csv(r"D:\桌面\311.csv")
    # df_m = pd.read_csv(r"D:\桌面\Active Learning\virtuals.csv")
    dfmul = StrToComposition(target_col_id='composition_obj').featurize_dataframe(df_m, 'composition', ignore_errors=True)

    feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                                              cf.ValenceOrbital(props=['avg']), cf.IonProperty(fast=True)])
    feature_labels = feature_calculators.feature_labels()
    result = feature_calculators.featurize_dataframe(dfmul, col_id='composition_obj')
    print(result.shape)
    result.to_csv("311DP.csv", index=None)
    # result.to_csv("virtuals_reasult.csv", index=None)