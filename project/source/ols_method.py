# coding=utf-8
"""
Ordinary least squares econometric model.

Statistical parameters inference and verification.

Copyright 2020 Karol Oleszek
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import chi2, t, f


def sgn(number):
    """Get unitarty sign number."""
    return -1 if number < 0 else (1 if number else 0)


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
    def __init__(self, x_data, y_data, verbose=True, alpha=0.05):
        """
        Initialize model with fit and statistical verification.

        :param x_data: explanatory variables (ndarray n x k).
        :param y_data: explained variable (ndarray n x 1).
        """
        self.x_data = x_data
        self.y_data = y_data
        self.n = x_data.shape[0]
        self.k = x_data.shape[1]
        self.params = np.zeros(self.k + 1)
        self.predictions = np.zeros(self.n)
        self.residuals = np.zeros(self.n)
        self.gram_schmidt = np.zeros((self.k, self.k))
        self.tests_passed = 0
        self.tests_total = 0
        self.verbose = verbose
        self.alpha = alpha
        self.r2_score = 0.0
        
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
        data = np.c_[np.ones(self.n), self.x_data]
        self.gram_schmidt = np.linalg.inv(data.T @ data)
        self.params = self.gram_schmidt @ data.T @ self.y_data
        predictions = self.predict(data)
        self.residuals = self.y_data - predictions

        y_mean = self.y_data.mean()
        y_mean_vec = np.repeat(y_mean, self.n)
        self.r2_score = ((predictions - y_mean_vec) ** 2).sum() / ((self.y_data - y_mean_vec) ** 2).sum()

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
        self.linearity_test()
        self.parameter_stability_test()
        self.residual_autocorrelation_test()
        self.log(f"Statistical verification: {self.tests_passed}/{self.tests_total} tests passed.")

    @statistical_test
    def coincidence_test(self):
        res = True
        for i in range(self.k):
            correlation = np.corrcoef(self.x_data[:, i], self.y_data)[0][1]
            param = self.params[i + 1]
            if not same_sgn(correlation, param):
                res = False
                break
        return res

    @statistical_test
    def catalizator_test(self):
        r_matrix = np.corrcoef(self.x_data.T)
        r0_vector = [np.corrcoef(self.x_data[:, i], self.y_data)[0][1] for i in range(self.k)]
        print(r_matrix, r0_vector)
        result = True
        for i in range(self.k):
            for j in range(self.k):
                if abs(r0_vector[j]) > abs(r0_vector[i]):
                    print(f'sprawdzam pare {i} {j}')
                    rij = r_matrix[i][j] * sgn(r0_vector[i]) * sgn(r0_vector[j])
                    ri = abs(r0_vector[i])
                    rj = abs(r0_vector[j])
                    try:
                        if rij > (ri / rj) or rij < 0:
                            result = False
                            print(f"Katalizator w parze {i} {j}")
                    except ZeroDivisionError:
                        pass
        return result

    @statistical_test
    def colinearity_test(self):
        for i in range(self.k):
            print(f"Sprawdzam wspoliniowosc zmiennej {i}")
            target = self.x_data[:, i].reshape(-1, 1)
            data = np.delete(self.x_data, i, axis=1)
            model = LinearRegression()
            model.fit(data, target)
            score = r2_score(target, model.predict(data))
            print(score)
            if score > 0.9:
                result = False
                print(f"Zmienna {i} wspoliniowa.")
        return False

    @statistical_test
    def residuals_normality_test(self):
        result = True
        n = self.n
        xm = self.residuals.mean()
        mean_vec = np.repeat(xm, n)
        u3 = (1 / n) * ((self.residuals - mean_vec) ** 3).sum()
        sig3 = ((1 / n) * ((self.residuals - mean_vec) ** 2).sum()) ** 1.5
        a = u3 / sig3
        u4 = (1 / n) * ((self.residuals - mean_vec) ** 4).sum()
        sig4 = ((1 / n) * ((self.residuals - mean_vec) ** 2).sum()) ** 2
        k = u4 / sig4
        jb = ((n - self.k) / 6) * (a ** 2 + 0.25 * ((k - 3) ** 2))
        chi2_alpha_2 = chi2.isf(df=2, q=self.alpha)
        print(jb, chi2_alpha_2)
        if jb > chi2_alpha_2:
            result = False
            print("Reszty nie mają rozkładu normalnego.")
        return result

    @statistical_test
    def parameters_significance_test(self):
        nk1 = (self.n - (self.k + 1))
        s2 = np.dot(self.residuals.T, self.residuals) / nk1
        da = self.gram_schmidt * s2
        print(da)
        t_alpha_nk1 = t.isf(df=nk1, q=self.alpha)
        result = True
        for i in range(self.k + 1):
            print(f"Testowanie istotnosci zmiennej {i}")
            t_stat = self.params[i] / da[i, i]
            print(t_stat, t_alpha_nk1)
            if abs(t_stat) < t_alpha_nk1:
                result = False
                print(f"Zmienna {i} nieistotna")
        return result

    @statistical_test
    def r2_significance_test(self):
        result = False
        f_alpha_r1r2 = f.isf(q=self.alpha, dfn=self.k, dfd=(self.n - (self.k + 1)))
        f_stat = (self.r2_score / (1 - self.r2_score)) * ((self.n - (self.k + 1)) / (self.k))
        print(f_stat, f_alpha_r1r2)
        if f_alpha_r1r2 < abs(f_stat):
            result = True
        else:
            print("Wspołczynnik r2 nieistotny")
        return result

    @statistical_test
    def homoskedacity_test(self):
        result = True
        n1 = self.n // 2
        n2 = self.n - n1
        r1 = n1 - (self.k + 1)
        r2 = n2 - (self.k + 1)
        f_alpha_r1r2 = f.isf(q=self.alpha, dfn=r1, dfd=r2)
        
        resid1 = self.residuals[:n1]
        resid2 = self.residuals[n1:]
        print(self.residuals, resid1, resid2)
        s12 = np.dot(resid1, resid1) / r1
        s22 = np.dot(resid2, resid2) / r2
        f_stat = s12 / s22
        print(f_stat, f_alpha_r1r2)
        if f_alpha_r1r2 < abs(f_stat):
            result = False
            print("Wystepuje heteroskedastycznosc")
        return result

    @statistical_test
    def linearity_test(self):
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
        [5, 3, 1],
        [6, 2, 3],
        [7, 3, 0],
        [8, 9, 0],
        [9, 2, 7],
        [10, 1, 1]
    ])
    y = np.array([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    ])
    model = OLS(x, y)
    print(model.params)
    print(model.residuals)
    print(model.r2_score)
