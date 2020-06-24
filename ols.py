# coding=utf-8
"""
Ordinary least squares.
"""


import numpy as np
from scipy.stats import t


def ols():
    """
    Perform OLS with steps.
    """
    n = int(input("Ilość wierszy: "))
    m = int(input("Ilość kolumn: "))
    matrix = []
    for _ in range(n):
        row = [int(x) for x in input().split(" ")]
        matrix.append(row)
    x = np.array(matrix)

    print("Macierz X wczytana. Podaj wektor Y:")
    y_row = [int(x) for x in input().split(" ")]
    y = np.array(y_row)
    a = np.linalg.inv(x.T @ x) @ x.T @ y
    y_hat = x @ a
    e = y - y_hat
    s_a = np.dot(e, e)
    r_2 = 1 - s_a / (np.dot(y, y) - n * (np.mean(y) ** 2))
    r_2_adj = r_2 - (1 - r_2) * (m) / (n - (m))
    s_2 = s_a / (n - (m))
    d_a = s_2 * np.linalg.inv(x.T @ x)
    param_erros = [np.sqrt(d_a[i, i]) for i in range(m)]
    rel_param_errors = [abs(param_erros[i] / a[i]) for i in range(m)]
    print("X")
    print(x)
    print("Y")
    print(y)
    print("X^TX")
    print(x.T @ x)
    print("X^TY")
    print(x.T @ y)
    print("(X^TX)-1")
    print(np.linalg.inv(x.T @ x))
    print("a")
    print(a)
    print("Y_hat")
    print(y_hat)
    print("e")
    print(e)
    print("s_a")
    print(s_a)
    print("r2")
    print(r_2)
    print("r2_adj")
    print(r_2_adj)
    print("s_2")
    print(s_2)
    print("d_a")
    print(d_a)
    print("param_errors")
    print(param_erros)
    print("rel_param_errors")
    print(rel_param_errors)
    print("Testing significance")
    param = int(input("Który testujemy:"))
    null_val = int(input("Null hypothesis value:"))
    t_a_i = (a[param] - null_val) / param_erros[param]
    print("statystyka t-stud")
    print(t_a_i)
    t_alpha_nk1 = t.isf(df=(n - m), q=float(input("significance level:")))
    print(t_alpha_nk1)
    print("istotnie różna:")
    print(abs(t_a_i) > t_alpha_nk1)




if __name__ == "__main__":
    ols()
