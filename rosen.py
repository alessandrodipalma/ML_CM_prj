import numpy as np
from numpy import transpose as t
from numpy.linalg import inv as inv, matrix_power
from scipy.optimize import line_search

from knapsack_secant import dai_fletch_a1
from line_search import backtracking_armijo_ls


class RosenGradientProjection:

    def __init__(self, Q, q, left_constr, right_constr, y, b, lam_low=1e-3, lam_upp=1, verbose=False):
        """

        :param Q: gram
        :param q:
        :param left_constr: left constraint vector
        :param right_constr:
        :param y: vector associated with the linear constraint s.t. y.T @ x = b
        :param b: vector associated with the linear constraint s.t. y.T @ x = b
        :param lam_low:
        :param lam_upp:
        """
        self.Q = Q
        self.Q_square = matrix_power(Q, 2)
        self.q = q

        self.f = lambda x: 0.5 * (x.T @ Q @ x) + q @ x
        self.df = lambda x: Q @ x + q

        self.left_constr = left_constr
        self.right_constr = right_constr

        self.n = self.Q.shape[1]
        self.I = np.identity(self.n)

        self.y = y
        self.b = b

        self.n_max = 10
        self.current_rule = 1
        self.lam_low = lam_low
        self.lam_upp = lam_upp
        self.rule_iter = 1

        n_half = int(self.n / 2)
        self.Y = np.append(np.ones(n_half), -np.ones(n_half))

        self.verbose = verbose

    def line_search(self, x, d):
        l_new = backtracking_armijo_ls(self.f, self.df, x, d)
        # print("lambda_opt={}".format(l_new))
        if l_new is not None:
            l = l_new
        if l_new < self.lam_low:
            l = self.lam_low
        if l_new > self.lam_upp:
            l = self.lam_upp

        return l

    def update_x(self, d, x):

        # lambda_opt, fc, gc, new_fval, old_fval, new_slope = line_search(self.f, self.df, x, d, amax=self.lam_upp,
        #                                                                 maxiter=100, c1=0.01, c2=0.9)
        # lambda_opt = armijo_wolfe_ls(lambda a: self.f(x + a * d), lambda a: self.df(x + a * d) @ d, lambda_max)
        # lambda_opt = backtracking_armijo_ls(lambda a: self.f(x + a * d), lambda a:  self.df(x + a * d) @ d, lambda_max)

        # print("lambda_opt={}, x={}, d={}".format(lambda_opt, x, d))

        lambda_opt = self.line_search(x, d)
        x = x + lambda_opt * d

        return x

    def _project(self, x):

        solver = dai_fletch_a1(self.left_constr, self.right_constr,
                               self.y, self.b, np.identity(self.n), x)
        xp = solver.solve(lam_i=1, d_lam=2)

        # solver.plot_xtory()
        return xp

    def solve(self, x0, max_iter=100, min_d=1e-3, x_opt=None, f_star=None):
        x = x0
        k = 1
        while k < max_iter:
            gradient = self.df(x)
            d = self._project(x - gradient) - gradient
            x = self.update_x(d, x)
            k += 1
            if np.linalg.norm(d) < min_d:
                break

        return x, d
