# coding=utf-8
"""
Dataset cleaning.

Econometrics course project.

Karol Oleszek 2020
"""

import pandas as pd

DATA_PATH = '2018_Financial_Data.csv'
OUT_PATH = 'clean_data.csv'


if __name__ == '__main__':
    print("Reading data...")
    data = pd.read_csv(DATA_PATH)

    print("Removing corrupted observations...")
    todel = []
    for i in range(len(data.index)) :
        if data.iloc[i].isnull().sum() >= 50:
            todel.append(i)
    data.drop(data.index[todel], inplace=True)

    print("Removing corrupted columns...")
    nacol = [i for i, x in enumerate(data.isnull().sum()) if x > 440]
    data.drop(data.columns[nacol], axis=1, inplace=True)

    print("Removing unneeded columns...")
    data.drop(['Unnamed: 0', 'Class'], axis=1, inplace=True)

    print("Renaming target column...")
    data.rename(columns={'2019 PRICE VAR [%]': 'PriceChange'}, inplace=True)

    print("Replacing missing values with column means...")
    data.fillna(data.mean(), inplace=True)

    print("One hot encoding sector column...")
    sectors = pd.get_dummies(data.Sector, drop_first=True)
    data.drop('Sector', axis=1, inplace=True)
    data = pd.concat([data, sectors], axis=1)

    print("Processing complete.")
    print(data)

    print("Saving results...")
    data.to_csv(OUT_PATH)
