# -*- coding: utf-8 -*-
"""
@Time : 2022/3/18
@Author : 王向东
@Email : wxd1872@163.com
@File : miaoshufu.py
@Project : main.py
"""



if __name__ == '__main__':
    import pandas as pd
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers import composition as cf
    from matminer.featurizers.conversions import StrToComposition

    df_m = pd.read_csv(r"D:\桌面\DP\f23.csv")
    df2 = StrToComposition(target_col_id='composition_obj').featurize_dataframe(df_m, 'composition', ignore_errors=True)
    feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                                              cf.ValenceOrbital(props=['avg']), cf.IonProperty(fast=True)])
    feature_labels = feature_calculators.feature_labels()
    result = feature_calculators.featurize_dataframe(df2, col_id='composition_obj')
    print(result.shape)
    result.to_csv("feature.csv", index=None)


