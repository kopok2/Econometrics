# coding=utf-8
"""
Ordinary least squares econometric model.

Statistical parameters inference and verification.

Copyright 2020 Karol Oleszek
"""

import numpy as np


def same_sgn(first, second):
    """Check whether both a and b have the same sign."""
    return (first < 0 and second < 0) or (first > 0 and second > 0)


def statistical_test(method):
    """
    Statistical test counter.
    :param method: test method.
    :return: wrapped method.
    """
    def wrapper(*args):
        """
        Perform statistical test success count.
        """
        args[0].tests_total += 1
        args[0].tests_passed += method(*args)
    return wrapper


class OLS:
    """Ordinary Least Squares econometric model."""
    def __init__(self, x_data, y_data, verbose=True):
        """
        Initialize model with fit and statistical verification.

        :param x_data: explanatory variables (ndarray n x k).
        :param y_data: explained variable (ndarray n x 1).
        """
        self.x_data = x_data
        self.y_data = y_data
        self.params = np.zeros(x_data.shape[1])
        self.predictions = np.zeros(y_data.shape[0])
        self.residuals = np.zeros(y_data.shape[0])
        self.tests_passed = 0
        self.tests_total = 0
        self.verbose = verbose
        self.run_model()

    def log(self, message):
        """
        Log model output.

        :param message: string to print (str).
        """
        if self.verbose:
            print(message)

    def run_model(self):
        """
        Run model inference and validation.
        """
        self.fit()
        self.validate()

    def fit(self):
        """
        Fit model parameters with initial data.
        """
        data = np.c_[np.ones(self.x_data.shape[0]), self.x_data]
        self.params = np.linalg.inv(data.T @ data) @ data.T @ self.y_data
        self.residuals = self.y_data - self.predict(data)

    def predict(self, x_data):
        """
        Infer predictions on x_data.

        :param x_data: explanatory variables (ndarray m x k).
        :return: predictions (ndarray m x 1).
        """
        return x_data @ self.params

    def validate(self):
        """
        Validate model statistically.
        """
        self.coincidence_test()
        self.catalizator_test()
        self.colinearity_test()
        self.residuals_normality_test()
        self.parameters_significance_test()
        self.r2_significance_test()
        self.homoskedacity_test()
        self.analitical_form_stability_test()
        self.parameter_stability_test()
        self.residual_autocorrelation_test()
        self.log(f"Statistical verification: {self.tests_passed}/{self.tests_total} tests passed.")

    @statistical_test
    def coincidence_test(self):
        res = True
        for i in range(self.x_data.shape[1]):
            correlation = np.corrcoef(self.x_data[:, i], self.y_data)[0][1]
            param = self.params[i + 1]
            if not same_sgn(correlation, param):
                res = False
                break
        return res

    @statistical_test
    def catalizator_test(self):
        r_matrix = np.corrcoef(self.x_data.T)
        r0_vector = [(i, np.corrcoef(self.x_data[:, i], self.y_data)[0][1]) for i in range(self.x_data.shape[1])]
        print(r_matrix, r0_vector)
        return False

    @statistical_test
    def colinearity_test(self):
        return False

    @statistical_test
    def residuals_normality_test(self):
        return False

    @statistical_test
    def parameters_significance_test(self):
        return False

    @statistical_test
    def r2_significance_test(self):
        return False

    @statistical_test
    def homoskedacity_test(self):
        return False

    @statistical_test
    def analitical_form_stability_test(self):
        return False

    @statistical_test
    def parameter_stability_test(self):
        return False

    @statistical_test
    def residual_autocorrelation_test(self):
        return False


if __name__ == '__main__':
    x = np.array([
        [1, 2, 3],
        [2, 2, 3],
        [3, 2, 4],
        [4, 2, 5],
        [5, 3, 1]
    ])
    y = np.array([
        1, 2, 3, 4, 5
    ])
    model = OLS(x, y)
    print(model.params)
    print(model.residuals)
