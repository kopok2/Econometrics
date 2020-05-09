# coding=utf-8
"""
Stock prices changes regression.

Econometrics course project.

Karol Oleszek 2020
"""

import pandas as pd

DATA_PATH = '2018_Financial_Data.csv'



if __name__ == '__main__':
    data = pd.read_csv(DATA_PATH)
    print(data)
    j = 0
    todel = []
    for i in range(len(data.index)) :
        if data.iloc[i].isnull().sum() >= 50:
            j += 1
            todel.append(i)
        print("Nan in row ", i , " : " ,  data.iloc[i].isnull().sum())
    print(todel)
    print(j)


    print(data.isnull().sum())
    data.drop(data.index[todel], inplace=True)
    nacol = [i for i, x in enumerate(data.isnull().sum()) if x > 440]
    print(len(nacol))
    print(nacol)
    data.drop(data.columns[nacol], axis=1, inplace=True)
    print(data)
    print([x for x in data.isnull().sum()])
    print(data.isnull().sum().sum())
