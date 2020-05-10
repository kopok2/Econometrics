# coding=utf-8
"""
Dataset exploration.

Econometrics course project.

Karol Oleszek 2020
"""

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = 'clean_data.csv'

if __name__ == '__main__':
    print("Reading data...")
    data = pd.read_csv(DATA_PATH)
    data.drop(data.columns[0], axis=1, inplace=True)
    print(data)

    print("Plotting correlation matrix...")
    f = plt.figure(figsize=(30, 26))
    plt.matshow(data.corr(), fignum=f.number)
    plt.xticks(range(data.shape[1]), data.columns, fontsize=4, rotation=90)
    plt.yticks(range(data.shape[1]), data.columns, fontsize=4)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.savefig("CorrelationMatrix.png", dpi=300)