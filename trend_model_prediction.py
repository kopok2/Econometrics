# coding=utf-8
"""
Trend model prediction error horizon.
"""

import numpy as np


def trend_model():
    """
    Trend model prediction error horizon.
    """
    a = np.array([int(x) for x in input('podaj parametry:').split(" ")])
    t = int(input("t: "))
    x_t_x = np.array([[t, sum([x for x in range(1, t + 1)])], [sum([x for x in range(1, t + 1)]), sum([x ** 2 for x in range(1, t + 1)])]])
    print("x_t_x")
    print(x_t_x)
    print("x_t_x-1")
    print(np.linalg.inv(x_t_x))
    s_2 = int(input("enter S^2: "))
    level = float(input("acceptable error level"))
    print("Spt = sqrt(s_2 (xtXtX-1x) + 1)")
    for i in range(1, int(input("Horizine")) + 1):
        print(i)
        s_p = np.sqrt(s_2 * ((np.dot(np.array([1, i]).T @ np.linalg.inv(x_t_x), np.array([1, i]))) + 1))
        y_pred = a[0] + i * a[1]
        print("pred: ", y_pred)
        print(s_p)
        print(abs(s_p / y_pred))
        print(abs(s_p / y_pred) < level)



if __name__ == "__main__":
    trend_model()
