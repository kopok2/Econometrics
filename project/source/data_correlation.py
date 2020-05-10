# coding=utf-8
"""
Dataset exploration.

Econometrics course project.

Karol Oleszek 2020
"""

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import os

DATA_PATH = 'clean_data.csv'

if __name__ == '__main__':
    tmp_var = open('variable_template.tex').read()
    print("Reading data...")
    data = pd.read_csv(DATA_PATH)
    data.drop(data.columns[0], axis=1, inplace=True)
    corr = data.corr()
    print(corr)
    print(corr['PriceChange'])
    stat = data['PriceChange'].describe()
    print(stat)
    vec = data['PriceChange'].to_numpy()
    mn = stat['25%'] - stat['std']
    mx = stat['75%'] + stat['std']
    plt.hist(vec, bins=200, range=(mn, mx))
    plt.savefig("YHistogram.png", dpi=300)
    plt.clf()
    plt.cla()

    col = corr['PriceChange']
    col.drop('PriceChange', axis=0, inplace=True)
    stat = col.describe()
    print(stat)
    vec = col.to_numpy()
    mn = stat['25%'] - stat['std']
    mx = stat['75%'] + stat['std']
    plt.hist(vec, bins=200, range=(mn, mx))
    plt.savefig("CorrHist.png", dpi=300)
    plt.clf()
    plt.cla()
    print(tmp_var.format(var_name='Corr',
                             mean=stat['mean'],
                             sd=stat['std'],
                             q1=stat['25%'],
                             q2=stat['50%'],
                             q3=stat['75%'],
                             min=stat['min'],
                             max=stat['max'],
                             var_source='CorrHist.png'
                             ))