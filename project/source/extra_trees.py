# coding=utf-8
"""
Extra trees regression.

Econometrics course project.

Karol Oleszek 2020
"""

import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
IN_DATA_PATH = 'clean_data.csv'

if __name__ == '__main__':
    data = pd.read_csv(IN_DATA_PATH)
    Y = data['PriceChange']
    X = data.drop('PriceChange', axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
    for mc in [ExtraTreesRegressor,
               LinearRegression,
               KNeighborsRegressor,
               GradientBoostingRegressor,
               MLPRegressor,
               AdaBoostRegressor,
               GaussianProcessRegressor,
               SGDRegressor,
               HistGradientBoostingRegressor]:
        print(mc)
        model = mc()
        print("Training model...")
        model.fit(X_train, Y_train)
        print("Training:")
        print(mean_absolute_error(Y_train, model.predict(X_train)))
        print("Testing:")
        print(mean_absolute_error(Y_test, model.predict(X_test)))