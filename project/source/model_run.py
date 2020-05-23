# coding=utf-8
"""
Run entire class of models and assess them.
"""

import numpy as np
import pandas as pd
import random
import math
from operator import itemgetter
from ols_method import OLS
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from numpy.linalg import LinAlgError
DATA_PATH = 'clean_data.csv'
LATEX_OUT_PATH = 'models.tex'
random.seed(0)



def transform_heteroskedacity(x_data, y_data, residuals):
    return x_data / np.sqrt(residuals ** 2), y_data / np.sqrt(residuals ** 2)


def transform_autocorrelation(x_data, y_data, residuals):
    target = residuals[1:].reshape(-1, 1)
    data = residuals[:-1].reshape(-1, 1)
    model = LinearRegression(fit_intercept=False)
    model.fit(data, target)
    rho = model.coef_[0][0]
    y_t = y_data[1:] - rho * y_data[:-1]
    x_t = x_data[1:, :] - rho * x_data[:-1, :]
    y1 = y_data[0] * math.sqrt(1 - rho ** 2)
    x1 = x_data[0, :] * math.sqrt(1 - rho ** 2)
    y = np.vstack((np.array(y1), y_t.reshape(-1, 1))).flatten()
    x = np.vstack((np.array(x1).reshape(1, -1), x_t))
    return x, y





if __name__ == '__main__':
    data = pd.read_csv(DATA_PATH)
    data.drop(data.columns[0], axis=1, inplace=True)
    target = data['PriceChange']
    data.drop('PriceChange', axis=1, inplace=True)

    for cl in data.columns:
        data[f"e^{{\\frac{{{cl} - \overline{{{cl}}}}}{{\\sigma_{{{cl}}}}}}}"] = np.exp((data[cl] - data[cl].mean()) / data[cl].std())
        if cl not in ["Communication Services", "Consumer Cyclical", "Consumer Defensive", "Energy", "Financial Services", "Healthcare", "Industrials", "Real Estate", "Technology", "Utilities"]:
            data[cl + "^2"] = data[cl] ** 2
            data[cl + "^3"] = data[cl] ** 3

    data, target = shuffle(data, target, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=0)
    x_train, x_test, y_train, y_test = x_train.copy(), x_test.copy(), y_train.copy(), y_test.copy()

    lt_out = open(LATEX_OUT_PATH, 'w')
    lt_out.write(open('models_template.tex').read())
    out_res = open('selected/raport.txt', 'w')
    i = 1
    print("Model|R2 Score|MAE Score|Correctness|Vars|Dropping|MaeCV|Mae POST")
    out_res.write("Model|R2 Score|MAE Score|Correctness|Vars|Dropping|MaeCV|Mae POST\n")
    print(mean_absolute_error(y_test, np.repeat(y_train.mean(), y_test.shape[0])))
    print(mean_absolute_error(y_train, np.repeat(y_train.mean(), y_train.shape[0])))
    results = []

    while x_train.shape[1] > 0:
        model_name = f"Model ekonometryczny z {x_train.shape[1]} zmiennymi"
        try:
            model = OLS(x_train.to_numpy(), y_train.to_numpy(), name=model_name, var_names=x_train.columns, verbose=False)
            model.run_model()
            mae_cv = -cross_val_score(LinearRegression(), x_train, y_train, cv=5, scoring='neg_mean_absolute_error').mean()
            pred = model.predict(np.c_[np.ones(x_test.shape[0]), x_test])
            mae_post = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_train, model.predict(np.c_[np.ones(x_train.shape[0]), x_train]))


            if model.r2_score > 1:
                raise LinAlgError
            print(f"Model {i}|{model.r2_score}|{model.mae_score}|{model.tests_passed}/{model.tests_total}|{len(model.params)}|{model.least_important_param_name}|{mae_cv}|{mae_post}")
            out_res.write(f"Model {i}|{model.r2_score}|{model.mae_score}|{model.tests_passed}/{model.tests_total}|{len(model.params)}|{model.least_important_param_name}|{mae_cv}|{mae_post}\n")
            results.append((f"Model {i}", model, model.r2_score, mae_cv, mae_post, model.var_names, model.tests_passed))

            x_auto, y_auto = transform_autocorrelation(x_train.to_numpy(), y_train.to_numpy(), model.residuals)
            model_auto = OLS(x_auto, y_auto, var_names=x_train.columns, verbose=False, name=f'Model {i} with autocorr')
            model_auto.run_model()
            mae_cv = -cross_val_score(LinearRegression(), x_auto, y_auto, cv=5, scoring='neg_mean_absolute_error').mean()
            pred = model_auto.predict(np.c_[np.ones(x_test.shape[0]), x_test])
            mae_post = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_auto, model_auto.predict(np.c_[np.ones(x_auto.shape[0]), x_auto]))
            if model_auto.r2_score > 1:
                raise LinAlgError
            print(f"Model {i} with autocorr|{model_auto.r2_score}|{model_auto.mae_score}|{model_auto.tests_passed}/{model_auto.tests_total}|{len(model_auto.params)}|{model_auto.least_important_param_name}|{mae_cv}|{mae_post}")
            out_res.write(f"Model {i}|{model_auto.r2_score}|{model_auto.mae_score}|{model_auto.tests_passed}/{model_auto.tests_total}|{len(model_auto.params)}|{model_auto.least_important_param_name}|{mae_cv}|{mae_post}\n")
            results.append((f"Model {i} with autocorr", model_auto, model_auto.r2_score, mae_cv, mae_post, model_auto.var_names, model_auto.tests_passed))

            x_hete, y_hete = transform_autocorrelation(x_train.to_numpy(), y_train.to_numpy(), model.residuals)
            model_hete = OLS(x_hete, y_hete, var_names=x_train.columns, verbose=False, name=f'Model {i} with hete')
            model_hete.run_model()
            mae_cv = -cross_val_score(LinearRegression(), x_hete, y_hete, cv=5,
                                      scoring='neg_mean_absolute_error').mean()
            pred = model_hete.predict(np.c_[np.ones(x_test.shape[0]), x_test])
            mae_post = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_hete, model_hete.predict(np.c_[np.ones(x_hete.shape[0]), x_hete]))
            if model_hete.r2_score > 1:
                raise LinAlgError
            print(
                f"Model {i} with heteroskedacity|{model_hete.r2_score}|{model_hete.mae_score}|{model_hete.tests_passed}/{model_hete.tests_total}|{len(model_hete.params)}|{model_hete.least_important_param_name}|{mae_cv}|{mae_post}")
            out_res.write(
                f"Model {i}|{model_hete.r2_score}|{model_hete.mae_score}|{model_hete.tests_passed}/{model_hete.tests_total}|{len(model_hete.params)}|{model_hete.least_important_param_name}|{mae_cv}|{mae_post}\n")
            results.append((f"Model {i} with heteroskedacity", model_hete, model_hete.r2_score, mae_cv, mae_post,
                            model_hete.var_names, model_hete.tests_passed))

            x_heau, y_heau = transform_autocorrelation(x_hete, y_hete, model_hete.residuals)
            model_heau = OLS(x_heau, y_heau, var_names=x_train.columns, verbose=False, name=f"Model {i} wiht both")
            model_heau.run_model()
            mae_cv = -cross_val_score(LinearRegression(), x_heau, y_heau, cv=5,
                                      scoring='neg_mean_absolute_error').mean()
            pred = model_heau.predict(np.c_[np.ones(x_test.shape[0]), x_test])
            mae_post = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_heau, model_heau.predict(np.c_[np.ones(x_heau.shape[0]), x_heau]))
            if model_heau.r2_score > 1:
                raise LinAlgError
            print(
                f"Model {i} with heteroautocorr|{model_heau.r2_score}|{model_heau.mae_score}|{model_heau.tests_passed}/{model_heau.tests_total}|{len(model_heau.params)}|{model_heau.least_important_param_name}|{mae_cv}|{mae_post}")
            out_res.write(
                f"Model {i}|{model_heau.r2_score}|{model_heau.mae_score}|{model_heau.tests_passed}/{model_heau.tests_total}|{len(model_heau.params)}|{model_heau.least_important_param_name}|{mae_cv}|{mae_post}\n")
            results.append((f"Model {i} with heteroautocorr", model_heau, model_heau.r2_score, mae_cv, mae_post,
                            model_heau.var_names, model_heau.tests_passed))

            x_train.drop(model.least_important_param_name, axis=1, inplace=True)
            x_test.drop(model.least_important_param_name, axis=1, inplace=True)

        except (LinAlgError, ZeroDivisionError):
            #model.colinearity_test()
            to_del = random.choice(model.var_names)
            x_train.drop(to_del, axis=1, inplace=True)
            x_test.drop(to_del, axis=1, inplace=True)
            print(f"Model {i}| niestabilny numerycznie | {to_del}")
            out_res.write(f"Model {i}| niestabilny numerycznie | {to_del}\n")
        i += 1
    results.sort(key=itemgetter(2), reverse=True)
    results[0][1].validate()
    print(results[0][1].latex_repr)
    lt_out.write('\\end{document}')
    open('selected/prognostic.tex', 'w').write(results[0][1].latex_repr)
