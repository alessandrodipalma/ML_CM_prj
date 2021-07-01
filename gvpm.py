import math
from copy import deepcopy

import numpy as np
from numpy.linalg import norm, inv, matrix_power
from scipy.optimize.linesearch import line_search
from matplotlib import pyplot as plt
from robinson import robinson
from gradientprojection import armijo_wolfe_ls, backtracking_armijo_ls
from knapsack_secant import dai_fletch_a1

class GVPM:
    """
    Solves a quadratic problem with box constraints using the Generalized Variable Projection Method.
    """

    def __init__(self, Q, q, left_constr, right_constr, y, b, a_min=1e-12, a_max=1e12,
                 n_min=3,
                 lam_low=1e-3, lam_upp=1):
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

        self.f = lambda x: 0.5 * ( x.T @ Q @ x ) + q @ x
        self.df = lambda x: Q @ x + q

        self.left_constr = left_constr
        self.right_constr = right_constr
        print("left={}\nright={}".format(left_constr,right_constr))
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

    # def line_search(self, x, d, l):
    #     l_new = self.lam_upp
    #     k = 0
    #     x = np.copy(x)
    #     while norm(d) > 1e-3:
    #         l_new = (d.T @ d) / (d.T @ self.Q @ d)
    #         d = self._project(x - l_new * self.df(x)) - x
    #         x += float(l_new) * d
    #         k += 1
    #
    #     if l_new is not None:
    #         l = l_new
    #     else:
    #         l = self.lam_upp
    #     if math.isnan(l):
    #         l = self.lam_low
    #     return l
    def line_search(self, x, d, l):
        l_new = backtracking_armijo_ls(self.f, self.df, x, d)
        # print("lambda_opt={}".format(l_new))
        if l_new is not None:
            l = l_new
        if l_new < self.lam_low:
            l = self.lam_low
        if l_new > self.lam_upp:
            l = self.lam_upp

        return l


    def _project(self, x):
        # xp = robinson(self.left_constr, self.right_constr, self.y, self.b).solve(x)
        solver = dai_fletch_a1(self.left_constr, self.right_constr,
                             self.y, self.b, np.identity(self.n), x)
        xp = solver.solve(lam_i = 1, d_lam= 2)
        # solver.plot_xtory()
        return xp



    def solve(self, x0, max_iter=30, min_d=1e-3):
        x = x0
        k = 0
        lam = 1
        # P = np.identity(self.n) - 1
        # print("g:{}\td={}\ta={}".format(norm(gradient), d,a))
        print(x)
        gradient = self.df(x)
        a = abs(1 / np.max(self._project(x - gradient) - x))
        gs = []
        ds = []
        while k == 0 or (norm(d) > min_d and k<max_iter):

            # print("before\t gradient={}\tx={}\ta={}".format(norm(gradient), norm(x), a))
            # print("K={}\tx={}".format(k, norm(x)))
            d = self._project(x - a * gradient) - x
            # print("projected d ={}".format(norm(d)))

            lam = self.line_search(x, d, lam)
            # print("lambda ", lam)

            x = x + lam * d
            gradient = self.df(x)
            print("gradient {}\tx={}\td={}\tlambda={}".format(norm(gradient), norm(x), norm(d), lam))
            if d.T @ self.Q @ d <= 0:
                print("amax")
                a = self.a_max
            else:
                a_new = self._select_updating_rule(d, a, lam)
                a = min(self.a_max, max(self.a_min, a_new))
            k += 1

            gs.append(norm(gradient))
            ds.append(norm(d))
        print("LAST K={}".format(k))
        print(norm(x), "  ", norm(gradient))

        # self.plot_gradient(gs, title="gradient norm descent")
        # self.plot_gradient(ds, title="projected gradient norm descent", color='r')
        # input()


        return x, d

    def plot_gradient(self, gradient_history, title, color='b'):
        plt.plot(range(0, len(gradient_history)), gradient_history, c=color)
        plt.title(title)
        plt.yscale('log')
        plt.show()
