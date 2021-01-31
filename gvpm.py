import numpy as np
from numpy.linalg import norm, inv
from scipy.optimize.linesearch import line_search


class GVPM:

    def __init__(self, f=None, df=None, Q=None, q=None, A=None, b=None, a_min=1e-3, a_max=1e30, m=10, gamma=1e-4, n_min=4,
                 lam_low=0.1, lam_upp=5):

        if Q is None:
            raise ValueError("either (Q,q) or (f,df) have to specified")
        self.f = lambda x: 0.5 * x.T @ Q @ x + q @ x
        self.df = lambda x: Q @ x + q
        self.n = self.Q.shape[0]
        self.a_min = a_min
        self.a_max = a_max
        self.m = m
        self.I = np.identity(self.n)
        self.Q = Q

        self.n_min = n_min

        self.current_rule = 1
        self.lam_low = lam_low
        self.lam_upp = lam_upp
        self.n = 1
        self.A = A
        self.b = b

    def update_rule(self, d, exp):
        return (d @ (self.Q ** (exp - 1)) @ d) / (d @ (self.Q ** exp) @ d)

    def select_updating_rule(self, d, a, lam):
        a_new = {1: self.update_rule(d, 1), 2: self.update_rule(d, 2)}

        if self.n > self.n_min or self.is_steplength_separator(a, a_new) or \
                self.is_bad_descent_generator(a, a_new[1], lam):
            if self.current_rule == 1:
                self.current_rule = 2
            else:
                self.current_rule = 1
            self.n = 1
        else:
            self.n += 1

        return a_new[self.current_rule]

    def is_bad_descent_generator(self, a, a1, lam):
        return (lam < self.lam_low and a == a1) or (lam > self.lam_upp and a == a1)

    def is_steplength_separator(self, a, a_new):
        return a_new[2] < a < a_new[1]

    def line_search(self, x, d, l):
        l_new = line_search(lambda a: self.f(x + a * d),
                            lambda a: d @ self.df(x + a * d), amax=1, amin=0)
        l_new = l_new[0]
        if self.lam_low <= l_new <= self.lam_low * l or l is None:
            l = l_new
        else:
            l = l / 2

        return l

    def project(self, x, gradient):
        I = self.I
        residual = self.A @ x - self.b
        active_ineq_constr = np.where(abs(residual) == 0)
        M = self.A[active_ineq_constr]

        P = I - M.T @ inv(M @ M.T) @ M
        d1 = - P @ gradient
        return d1

    def solve(self, x0, max_iter=100):
        x = x0

        gradient = self.df(x)
        a = 1 / np.max(self.project(x - gradient) - x)
        P = self.project(x - a * gradient) - x

        k = 0
        lam = 0
        while np.max(P) > 1e-5 and k < max_iter:
            d = self.project(x - a * gradient) - x
            lam = line_search(x, d, lam)
            x = x + lam * d

            if d.t @ self.Q @ d <= 0:
                a = self.a_max
            else:
                a_new = self.select_updating_rule(d, a, lam)
                a = min(self.a_max, max(self.a_min, a_new))
            k += 1

        return x
