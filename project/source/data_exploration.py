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
    print("Reading data...")
    data = pd.read_csv(DATA_PATH)
    data.drop(data.columns[0], axis=1, inplace=True)
    data['PriceChange'].hist(bins=1000)
    plt.savefig("YHistogram.png", dpi=300)
    print(data)

    print("Plotting correlation matrix...")
    f = plt.figure(figsize=(30, 26))
    plt.matshow(data.corr(), fignum=f.number)
    plt.xticks(range(data.shape[1]), data.columns, fontsize=4, rotation=90)
    plt.yticks(range(data.shape[1]), data.columns, fontsize=4)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.savefig("CorrelationMatrix.png", dpi=300)
    tmp_vars = open('variables_template.tex').read()
    tmp_var = open('variable_template.tex').read()
    out = ''
    for col in data.columns:
        if col == 'PriceChange':
            continue
        print(col)
        stat = data[col].describe()
        print(stat)
        out += tmp_var.format(var_name=col,
                             mean=stat['mean'],
                             sd=stat['std'],
                             q1=stat['25%'],
                             q2=stat['50%'],
                             q3=stat['75%'],
                             min=stat['min'],
                             max=stat['max']
                             )
        vec = data[col].to_numpy()
        fig = plt.hist(vec, bins=100)
        plt.savefig("variables/" + col + ".png", dpi=300)
        plt.clf()
    out_tex = tmp_vars.format(vars=out)
    open("variables.tex", "w").write(out_tex)
    os.system('pdflatex variables')