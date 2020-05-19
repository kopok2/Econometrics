# coding=utf-8
"""
Ordinary least squares econometric model.

Statistical parameters inference and verification.

Copyright 2020 Karol Oleszek
"""

import math
import numpy as np
from operator import itemgetter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import chi2, t, f, norm


def sgn(number):
    """Get unitary sign number."""
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
    def __init__(self, x_data, y_data, verbose=True, alpha=0.05, name='Econometric Model', var_names=None):
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
        self.name = name
        self.var_names = var_names
        self.latex_repr = f'\\section{{{self.name}}}\n'
        self.make_latex_model()
        self.r2_score = 0.0
        self.mae_score = 0.0
        
        self.run_model()

    def ltw(self, msg):
        """
        Write to model latex representation.

        :param:
        msg: text to be added to latex (str).
        """
        self.latex_repr += msg

    def make_latex_model(self):
        """
        Write down model formula in latex.
        """
        self.ltw('\\subsection{Postać modelu}\n')
        model_form = " + ".join((f'\\alpha_{i+1}{var_name}' for i, var_name in enumerate(self.var_names)))
        self.ltw(f'\\[ \hat{{Y}} = \\alpha_0 + {model_form}\\]')

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
        self.mae_score = abs(self.residuals).mean()
        self.ltw('\\subsection{Wyestymowane parametry modelu}\n')
        for i, param in enumerate(self.params):
            self.ltw(f'\\[\\alpha_{i} = {param}\\]\n')
        self.ltw('\\subsection{Wskaźniki jakości modelu}\n')
        self.ltw(f'Współczynnik determinacji ~$R^2 = {self.r2_score}$\n\n')
        self.ltw(f'Średni absolutny błąd prognozy \\textit{{ex post}} ~$MAE = {self.mae_score}$\n')

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
        self.ltw('\\subsection{Weryfikacja poprawności modelu}\n')
        self.ltw(f'Weryfikacja poprawności modelu przy poziomie istotności ~$\\alpha = {self.alpha}$\n')
        self.coincidence_test()
        self.catalizator_test()
        self.colinearity_test()
        self.residuals_normality_test()
        self.parameters_significance_test()
        self.r2_significance_test()
        self.linearity_test()
        self.parameter_stability_test()
        self.homoskedacity_test()
        self.residual_autocorrelation_test()
        self.ltw('\\subsubsection{Wynik weryfikacji poprawności modelu}\n')
        self.ltw(f'{self.tests_passed}/{self.tests_total} testów poprawności modelu dało wynik pozytywny. ')
        if self.tests_passed == self.tests_total:
            self.ltw('Model jest poprawny.\n')
        else:
            self.ltw('Model nie jest poprawny.\n')
        self.log(f"Statistical verification: {self.tests_passed}/{self.tests_total} tests passed.")

    @statistical_test
    def coincidence_test(self):
        result = True
        self.ltw('\\subsubsection{Koincydencja}\n')
        for i in range(self.k):
            correlation = np.corrcoef(self.x_data[:, i], self.y_data)[0][1]
            param = self.params[i + 1]
            self.ltw(f'\\[sgn(\\alpha_{i + 1}) = {sgn(param)}\\]\n')
            self.ltw(f'\\[sgn(r_{i + 1}) = {sgn(correlation)}\\]\n')
            if not same_sgn(correlation, param):
                result = False
                print('brak koincydencji', correlation, param, sgn(correlation), sgn(param))
                self.ltw('Brak koincydencji.\n')
            else:
                self.ltw('Koincydencja.\n')

        return result

    @statistical_test
    def catalizator_test(self):
        r_matrix = np.corrcoef(self.x_data.T)
        r0_vector = [np.corrcoef(self.x_data[:, i], self.y_data)[0][1] for i in range(self.k)]
        print(r_matrix, r0_vector)
        result = True
        self.ltw('\\subsubsection{Katalizatory}\n')
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
                            self.ltw(f'Zmienna ~$X_{i + 1}$ jest katalizatorem w parze (~$X_{i+1}$, ~$X_{j + 1}$)\n')
                            print(f"Katalizator w parze {i} {j}")
                    except ZeroDivisionError:
                        pass
        if result:
            self.ltw('Brak katalizatorów.\n')
        return result

    @statistical_test
    def colinearity_test(self):
        result = True
        self.ltw('\\subsubsection{Współliniowość zmiennych}\n')
        for i in range(self.k):
            print(f"Sprawdzam wspoliniowosc zmiennej {i}")
            target = self.x_data[:, i].reshape(-1, 1)
            data = np.delete(self.x_data, i, axis=1)
            model = LinearRegression()
            model.fit(data, target)
            score = r2_score(target, model.predict(data))
            print(score)
            self.ltw(f'Zmienna ~$X_{i + 1}$ w zależności od reszty zmiennych - ~$R^2 = {score}$.\n')
            if score > 0.9:
                result = False
                self.ltw('Występuje współliniowość.\n\n')
                print(f"Zmienna {i} wspoliniowa.")
            else:
                self.ltw('Nie występuje współliniowość.\n\n')
        return result

    @statistical_test
    def residuals_normality_test(self):
        result = True
        self.ltw('\\subsubsection{Normalność rozkładu reszt}\n')
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
        self.ltw(f'\\[JB = {jb}\\]\n')
        self.ltw(f'\\[\chi^2_{{{self.alpha}, 2}} = {chi2_alpha_2}\\]\n')
        if jb > chi2_alpha_2:
            result = False
            self.ltw('Reszty nie mają rozkładu normalnego.\n')
            print("Reszty nie mają rozkładu normalnego.")
        else:
            self.ltw('Reszty mają rozkład normalny.\n')
        return result

    @statistical_test
    def parameters_significance_test(self):
        nk1 = (self.n - (self.k + 1))
        s2 = np.dot(self.residuals.T, self.residuals) / nk1
        da = self.gram_schmidt * s2
        print(da)
        t_alpha_nk1 = t.isf(df=nk1, q=self.alpha)
        result = True
        self.ltw('\\subsubsection{Istotność zmiennych objaśniających}\n')
        for i in range(self.k + 1):
            print(f"Testowanie istotnosci zmiennej {i}")
            t_stat = self.params[i] / da[i, i]
            print(t_stat, t_alpha_nk1)
            self.ltw(f'\\[t_{{\\alpha_{i + 1}}} = {t_stat}\\]\n')
            self.ltw(f'\\[t_{{{self.alpha}, {nk1}}} = {t_alpha_nk1}\\]\n')
            if abs(t_stat) < t_alpha_nk1:
                result = False
                self.ltw(f'Zmienna ~$X_{i + 1}$ jest statystycznie nieistotna.\n')
                print(f"Zmienna {i} nieistotna")
            else:
                self.ltw(f'Zmienna ~$X_{i + 1}$ jest statystycznie istotna.\n')
        return result

    @statistical_test
    def r2_significance_test(self):
        result = False
        self.ltw('\\subsubsection{Istotność współczynnika determinacji}\n')
        f_alpha_r1r2 = f.isf(q=self.alpha, dfn=self.k, dfd=(self.n - (self.k + 1)))
        f_stat = (self.r2_score / (1 - self.r2_score)) * ((self.n - (self.k + 1)) / (self.k))
        print(f_stat, f_alpha_r1r2)
        self.ltw(f'\\[F = {f_stat}\\]\n')
        self.ltw(f'\\[F_{{{self.alpha}, {self.k}, {self.n - (self.k + 1)}}} = {f_alpha_r1r2}\\]\n')
        if f_alpha_r1r2 < abs(f_stat):
            result = True
            self.ltw('Współczynnik determinacji ~$R^2$ jest statystycznie istotny.\n')
        else:
            print("Wspołczynnik r2 nieistotny")
            self.ltw('Współczynnik determinacji ~$R^2$ jest statystycznie nieistotny.\n')
        return result

    @statistical_test
    def homoskedacity_test(self):
        result = True
        self.ltw('\\subsubsection{Homoskedastyczność}\n')
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
        self.ltw(f'\\[F = {f_stat}\\]\n')
        self.ltw(f'\\[F_{{{self.alpha}, {r1}, {r2}}} = {f_alpha_r1r2}\\]\n')
        if f_alpha_r1r2 < abs(f_stat):
            result = False
            print("Wystepuje heteroskedastycznosc")
            self.ltw('W modelu występuje heteroskedastyczność.\n')
        else:
            self.ltw('Model jest homoskedastyczny.\n')
        return result

    @statistical_test
    def linearity_test(self):
        result = True
        self.ltw('\\subsubsection{Liniowość postaci modelu}\n')
        pairs = [(resid, y) for resid, y in zip(self.residuals, self.y_data)]
        pairs.sort(key=itemgetter(1))
        n1 = 0
        n2 = 0
        r = 0
        norm_alpha = norm.isf(q=self.alpha)
        last = 0
        for pair in pairs:
            if pair[0] > 0:
                n1 += 1
            elif pair[0] < 0:
                n2 += 1
            if pair[0] > 0 and not last > 0:
                r += 1
            elif pair[0] < 0 and not last < 0:
                r += 1
            last = pair[0]
        n = self.n
        z = (r - (((2 * n1 * n2) / n) + 1)) / (math.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n)) / ((n-1) * (n ** 2))))
        print(z, norm_alpha)
        self.ltw(f'\\[Z = {z}\\]\n')
        self.ltw(f'\\[k_{{{self.alpha}, 0, 1}} = {norm_alpha}\\]\n')
        if abs(z) > norm_alpha:
            result = False
            print("Model nieliniowy.")
            self.ltw("Postać modelu nie jest liniowa.\n")
        else:
            self.ltw('Postać modelu jest liniowa.\n')
        return result

    @statistical_test
    def parameter_stability_test(self):
        result = True
        self.ltw('\\subsubsection{Stabilność parametrów modelu}\n')
        n1 = self.n // 2
        n2 = self.n - n1
        r1 = self.k + 1
        r2 = self.n - 2 * (self.k + 1)
        f_alpha_r1r2 = f.isf(q=self.alpha, dfn=r1, dfd=r2)
        
        data1 = self.x_data[:n1,:]
        target1 = self.y_data[:n1].reshape(-1, 1)
        data2 = self.x_data[n1:,:]
        target2 = self.y_data[n1:].reshape(-1, 1)
        model1 = LinearRegression()
        model1.fit(data1, target1)
        pred1 = model1.predict(data1)
        model2 = LinearRegression()
        model2.fit(data2, target2)
        pred2 = model2.predict(data2)

        resid1 = self.y_data[:n1] - pred1[:, 0]
        resid2 = self.y_data[n1:] - pred2[:, 0]
        rsk = np.dot(self.residuals, self.residuals)
        rsk1 = np.dot(resid1, resid1)
        rsk2 = np.dot(resid2, resid2)
        f_stat = ((rsk - (rsk1 + rsk2)) / (rsk1 + rsk2)) * (r2 / r1)
        self.ltw(f'\\[F = {f_stat}\\]\n')
        self.ltw(f'\\[F_{{{self.alpha}, {r1}, {r2}}} = {f_alpha_r1r2}\\]\n')
        print(f_stat, f_alpha_r1r2)
        if f_alpha_r1r2 < abs(f_stat):
            result = False
            print("Parametry modelu nie są stabilne.")
            self.ltw("Parametry modelu nie są stabilne.\n")
        else:
            self.ltw('Parametry modelu są stabilne.\n')
        return result

    @statistical_test
    def residual_autocorrelation_test(self):
        result = True
        self.ltw('\\subsubsection{Autokorelacja czynnika losowego I rzędu}\n')
        target = self.residuals[1:].reshape(-1, 1)
        data = np.c_[self.x_data[1:, :], self.residuals[:-1]]
        model = LinearRegression()
        model.fit(data, target)
        score = r2_score(target, model.predict(data))
        print(score)
        lm = (self.n - 1) * score
        chi2_alpha_1 = chi2.isf(q=self.alpha, df=1)
        print(lm, chi2_alpha_1)
        self.ltw(f'\\[LM = {lm}\\]\n')
        self.ltw(f'\\[\\chi^2_{{{self.alpha}, 1}} = {chi2_alpha_1}\\]\n')
        if lm > chi2_alpha_1:
            result = False
            self.ltw('W modelu występuje autokorelacja czynnika losowego I rzędu.\n')
            print("W modelu występuje autokorelacja pierwszego rzedu.")
        else:
            self.ltw('W modelu nie występuje autokorelacja czynnika losowego I rzędu.\n')

        return result


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
        [11, 1, 1]
    ])
    y = np.array([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    ])
    model = OLS(x, y, var_names=["X_1", "X_2", "X_3"])
    print(model.latex_repr)
    print(model.params)
    print(model.residuals)
    print(model.r2_score)
