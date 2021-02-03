from copy import deepcopy

import numpy as np
from numpy.linalg import norm, inv, matrix_power
from scipy.optimize.linesearch import line_search


class GVPM:

    def __init__(self, Q, q, A=None, b=None, a_min=1e-30, a_max=1e30,
                 n_min=2,
                 lam_low=0.1, lam_upp=1):
        self.Q = Q
        self.Q_square = matrix_power(Q,2)
        self.q = q

        self.f = lambda x: 0.5 * x.T @ Q @ x + q@x
        self.df = lambda x: Q @ x + q

        self.A = A
        self.b = b
        self.n = self.A.shape[1]
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

        l_new = line_search(self.f, self.df, x, d, amax=self.lam_upp, amin=self.lam_low)
        l_new = l_new[0]
        print("lambda_opt={}".format(l_new))
        if l_new is not None:
            l = l_new
        else:
            l = l / 2
            if l < self.lam_low:
                l = self.lam_low
            elif l > self.lam_upp:
                l = self.lam_upp

        return l

    def project(self, x):
        I = self.I
        residual = self.A @ x - self.b
        active_ineq_constr = np.where(abs(residual) <= 1e-6)
        M = self.A[active_ineq_constr]
        # M = np.concatenate((M, np.append(np.ones(int(self.n/2)), -np.ones(int(self.n/2))).reshape((1,self.n))))
        # print(M)
        # if np.all((x) == 0):
        #     M = M[:int(M.shape[0] / 2)]
        P = I - M.T @ inv(M @ M.T) @ M

        # print("M:{}\nP={}\ng={}".format(M, P,  gradient))
        d1 = - P @ x

        return d1

    def solve(self, x0, max_iter=100):
        x = x0
        k = 0
        lam = 1
        # print("g:{}\td={}\ta={}".format(norm(gradient), d,a))
        gradient = self.df(x)
        a = 1 / norm(gradient)
        xs = [] # history of x values
        gs = []  # history of gradient values
        while k == 0 or (np.max(d) > 1e-6 and k < max_iter):

            d = self.project(x - a * gradient) - x
            lam = self.line_search(x, d, lam)
            print("K={}\tx:{}\tg:{}\ta:{}\td:{}\tlambda:{}".format(k, norm(x), norm(gradient), a, norm(d), lam))
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

        return x
