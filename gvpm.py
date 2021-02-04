from copy import deepcopy

import numpy as np
from numpy.linalg import norm, inv, matrix_power
from scipy.optimize.linesearch import line_search
from matplotlib import pyplot as plt

from utils import plot_error


class GVPM:

    def __init__(self, Q, q, left_constr, right_constr, a_min=1e-30, a_max=1e30,
                 n_min=3,
                 lam_low=0.1, lam_upp=1):
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

        self.n_min = n_min
        self.n_max = 10
        self.current_rule = 1
        self.lam_low = lam_low
        self.lam_upp = lam_upp
        self.rule_iter = 1

    def update_rule_1(self, d):
        return (d.T @ d) / (d.T @ self.Q @ d)

    def update_rule_2(self, d):
        return (d.T @ self.Q @ d) / (d.T @ self.Q_square @ d)

    def select_updating_rule(self, d, a, lam):
        a_new = {1: self.update_rule_1(d), 2: self.update_rule_2(d)}

        if self.rule_iter > self.n_min:
            if self.rule_iter > self.n_max or self.is_steplength_separator(a, a_new) or \
                    self.is_bad_descent_generator(a, a_new[1], lam):
                if self.current_rule == 1:
                    self.current_rule = 2
                else:
                    self.current_rule = 1
                self.rule_iter = 1
                print(" --- switch")

        else:
            self.rule_iter += 1

        return a_new[self.current_rule]

    def is_bad_descent_generator(self, a, a1, lam):
        return (lam < self.lam_low and a == a1) or (lam > self.lam_upp and a == a1)

    def is_steplength_separator(self, a, a_new):
        # print("a1: {}\ta2: {}".format(a_new[1], a_new[2]))
        return a_new[2] < a < a_new[1]

    def line_search(self, x, d, l):

        l_new = line_search(self.f, self.df, x, d, amax=self.lam_upp)
        l_new = l_new[0]
        print("lambda_opt={}".format(l_new))
        if l_new is not None:
            l = l_new
        else:
            # l = l / 2
            # if l < self.lam_low:
            #     l = self.lam_low
            # elif l > self.lam_upp:
            #     l = self.lam_upp
            l = self.lam_upp

        return l

    def project(self, x):

        d = np.zeros(x.shape)
        for i in range(len(x)):
            d[i] = max(self.left_constr[i], min(x[i], self.right_constr[i]))

        n_half = int(len(x) / 2)
        M = np.append(np.ones(n_half), -np.ones(n_half)).reshape((1, self.n))
        # P = M.T @ inv(M @ M.T) @ M
        # d = P @ d
        if M @ d != 0:
            print("CONSTRAINT UNSATISFIED")

        # print("\n\nprojected a*:{}\n projected a:{}".format(d[:n_half],d[n_half:]))
        return d

    def one_side(self, d):
        n_half = int(len(d) / 2)
        for i in range(n_half):
            if d[i] * d[i+n_half] != 0:
                if d[i] > d[i+n_half]:
                    d[i+n_half] = 0
                else:
                    d[i] = 0
        return d

    def solve(self, x0, max_iter=100):
        x = x0
        k = 0
        lam = 0.5
        # print("g:{}\td={}\ta={}".format(norm(gradient), d,a))
        gradient = self.df(x)
        a = 1 / np.max(self.project(x - gradient) - x)

        xs = []  # history of x values
        gs = []  # history of gradient values
        while k == 0 or (np.max(d) > 1e-8 and k < max_iter):

            d = self.project(x - a * gradient) - x
            lam = self.line_search(x, d, lam)
            n_half = int(x.shape[0]/2)
            # # input("{}\n{}".format(x[:n_half],x[n_half:]))
            # for i in range(n_half):
            #     if x[i] * x[i + n_half] != 0:
            #         print("multipliers at position {} are infeasible".format(i))

            print("\n\nK={}\tx:{}\tg:{}\ta:{}\td:{}\tlambda:{}\n".format(k, norm(x), norm(gradient), a, norm(d), lam))
            print("\n\na*:{}\n a:{}".format(x[:n_half], x[n_half:]))
            print("\n\nd*:{}\n projected d:{}".format(d[:n_half], d[n_half:]))

            xs.append(deepcopy(x))
            gs.append(norm(gradient))
            x = x + lam * d

            gradient = self.df(x)

            if d.T @ self.Q @ d <= 0 and k > 1:
                print("amax")
                a = self.a_max
                # return x
            else:
                a_new = self.select_updating_rule(d, a, lam)
                a = min(self.a_max, max(self.a_min, a_new))
            k += 1

        # self.plot_gradient(gs)

        return x

    def plot_gradient(self, gradient_history):
        plt.plot(range(0, len(gradient_history)), gradient_history, c='b')
        plt.title('gradient norm descent')
        plt.show()
