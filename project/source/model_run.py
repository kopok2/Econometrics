# coding=utf-8
"""
Run entire class of models and assess them.
"""

import numpy as np
import pandas as pd
from ols_method import OLS

DATA_PATH = 'clean_data.csv'
LATEX_OUT_PATH = 'models.tex'

if __name__ == '__main__':
    data = pd.read_csv(DATA_PATH)
    data.drop(data.columns[0], axis=1, inplace=True)
    target = data['PriceChange']
    data.drop('PriceChange', axis=1, inplace=True)

    for cl in data.columns:
        data[f"e^{{\\frac{{{cl} - \overline{{{cl}}}}}{{\\sigma_{{{cl}}}}}"] = np.exp((data[cl] - data[cl].mean()) / data[cl].std())
        data[cl + "^2"] = data[cl] ** 2
        data[cl + "^3"] = data[cl] ** 3


    lt_out = open(LATEX_OUT_PATH, 'w')
    lt_out.write(open('models_template.tex').read())

    while data.shape[1] > 1:
        print(data.shape[1])
        for cl in data.columns:
            print(cl)
        model_name = f"Model ekonometryczny z {data.shape[1]} zmiennymi"
        model = OLS(data.to_numpy(), target.to_numpy(), name=model_name, var_names=data.columns)
        lt_out.write(model.latex_repr)
        data.drop(model.least_important_param_name, axis=1, inplace=True)
    lt_out.write('\\end{document}')