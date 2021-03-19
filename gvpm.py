from copy import deepcopy

import numpy as np
from numpy.linalg import norm, inv, matrix_power
from scipy.optimize.linesearch import line_search
from matplotlib import pyplot as plt

from gradientprojection import armijo_wolfe_ls, backtracking_armijo_ls
from knapsack_secant import dai_fletch_a1

class GVPM:
    """
    Solves a quadratic problem with box constraints using the Generalized Variable Projection Method.
    """

    def __init__(self, Q, q, left_constr, right_constr, y, b, a_min=1e-30, a_max=1e30,
                 n_min=3,
                 lam_low=0.1, lam_upp=1):
        """

        :param Q: gram
        :param q:
        :param left_constr: left constraint vector
        :param right_constr:
        :param y: vector associated with the linear constraint s.t. y.T @ x = b
        :param b: vector associated with the linear constraint s.t. y.T @ x = b
        :param a_min:
        :param a_max:
        :param n_min:
        :param lam_low:
        :param lam_upp:
        """
        self.Q = Q
        self.Q_square = matrix_power(Q, 2)
        self.q = q

        self.f = lambda x: 0.5 * x.T @ Q @ x + q @ x
        self.df = lambda x: Q @ x + q

        self.left_constr = left_constr
        self.right_constr = right_constr
        self.n = self.Q.shape[1]
        self.a_min = a_min
        self.a_max = a_max
        self.I = np.identity(self.n)

        self.y = y
        self.b = b

        self.n_min = n_min
        self.n_max = 10
        self.current_rule = 1
        self.lam_low = lam_low
        self.lam_upp = lam_upp
        self.rule_iter = 1

        n_half = int(self.n / 2)
        self.Y = np.append(np.ones(n_half), -np.ones(n_half))

    def _update_rule_1(self, d):
        return (d.T @ d) / (d.T @ self.Q @ d)

    def _update_rule_2(self, d):
        return (d.T @ self.Q @ d) / (d.T @ self.Q_square @ d)

    def _select_updating_rule(self, d, a, lam):
        a_new = {1: self._update_rule_1(d), 2: self._update_rule_2(d)}

        if self.rule_iter > self.n_min:
            if self.rule_iter > self.n_max or self._is_steplength_separator(a, a_new) or \
                    self._is_bad_descent_generator(a, a_new[1], lam):
                if self.current_rule == 1:
                    self.current_rule = 2
                else:
                    self.current_rule = 1
                self.rule_iter = 1
                # print(" --- switch")

        else:
            self.rule_iter += 1

        return a_new[self.current_rule]

    def _is_bad_descent_generator(self, a, a1, lam):
        return (lam < self.lam_low and a == a1) or (lam > self.lam_upp and a == a1)

    def _is_steplength_separator(self, a, a_new):
        return a_new[2] < a < a_new[1]

    def line_search(self, x, d, l):
        l_new = self.lam_upp
        k = 0
        while norm(d) > 1e-3:
            l_new = (d.T @ d) / (d.T @ self.Q @ d)
            d = self._project(x - l_new * self.df(x)) - x
            x += l_new * d
            k += 1

        if l_new is not None:
            l = l_new
        else:
            l = self.lam_upp

        return l

    def _project(self, x):
        return dai_fletch_a1(self.left_constr, self.right_constr,
                             self.y, self.b, np.identity(self.n), x).solve()



    def solve(self, x0, max_iter=100, min_d=1e-5):
        x = x0
        k = 0
        lam = 0.5
        # P = np.identity(self.n) - 1
        # print("g:{}\td={}\ta={}".format(norm(gradient), d,a))
        gradient = self.df(x)
        a = 1 / np.max(self._project(x - gradient) - x)

        while k == 0 or (np.max(d) > min_d and k < max_iter):

            print(gradient)
            print("K={}".format(k))
            d = self._project(x - a * gradient) - x
            print("projected d ={}".format(norm(d)))
            # d = P @ d

            # if not np.all(x == 0):
            # M = np.append(np.ones(self.n), -np.ones(self.n))

            # P = np.identity(self.n) - (M.T @ M) / (M @ M.T)

            # print("M@M.T = {}\tM.T@M={}\td={}".format((M.T @ M), (M @ M.T), 0))

            lam = self.line_search(x, d, lam)

            # print("\n\nK={}\tx:{}\tg:{}\ta:{}\td:{}\tlambda:{}\n".format(k, norm(x), norm(gradient), a, norm(d), lam))

            x = x + lam * d
            gradient = self.df(x)

            if d.T @ self.Q @ d <= 0:
                print("amax")
                a = self.a_max
            else:
                a_new = self._select_updating_rule(d, a, lam)
                a = min(self.a_max, max(self.a_min, a_new))
            k += 1

        # self.plot_gradient(gs)
        print("K={}".format(k))
        return x

    def plot_gradient(self, gradient_history):
        plt.plot(range(0, len(gradient_history)), gradient_history, c='b')
        plt.title('gradient norm descent')
        plt.show()
