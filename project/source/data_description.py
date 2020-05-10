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

    tmp_vars = open('variables_template.tex').read()
    tmp_var = open('variable_template.tex').read()
    out = ''
    for col in data.columns:
        if col == 'PriceChange':
            continue
        print(col)
        stat = data[col].describe()
        print(stat)
        png_name = col.replace('/', '_').replace('&', '_')
        out += tmp_var.format(var_name=col.replace('&', 'and'),
                             mean=stat['mean'],
                             sd=stat['std'],
                             q1=stat['25%'],
                             q2=stat['50%'],
                             q3=stat['75%'],
                             min=stat['min'],
                             max=stat['max'],
                             var_source=png_name
                             )
        vec = data[col].to_numpy()
        mn = stat['25%'] - stat['std']
        mx = stat['75%'] + stat['std']
        plt.hist(vec, bins=200, range=(mn, mx))
        plt.savefig("variables/" + png_name + ".png", dpi=300)
        plt.clf()
        plt.cla()
    out_tex = tmp_vars.format(vars=out)
    open("var.tex", "w").write(out_tex)
    os.system('pdflatex var')