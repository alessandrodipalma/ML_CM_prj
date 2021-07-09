import time

import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm, matrix_power

from line_search import backtracking_armijo_ls
from knapsack_secant import dai_fletch_a1


class GVPM:
    """
    Solves a quadratic problem with box constraints using the Generalized Variable Projection Method.
    """

    def __init__(self, Q, q, left_constr, right_constr, y, b, a_min=1e-8, a_max=1e8,
                 n_min=1,
                 lam_low=1e-3, lam_upp=1, verbose=False, proj_tol=1e-8, plots=False):
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

        self.f = lambda x: 0.5 * (x.T @ Q @ x) + q @ x
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
        self.projection_tol = proj_tol
        self.verbose = verbose

        self.plots = plots

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
    #     while norm(d) > 1e-6:
    #         l_new = (d.T @ d) / (d.T @ self.Q @ d)
    #         d = self._project(x - l_new * self.df(x)) - x
    #         x += float(l_new) * d
    #         k += 1
    #
    #     if l_new is not None:
    #         l = l_new
    #     if l_new < self.lam_low:
    #         l = self.lam_low
    #     if l_new > self.lam_upp:
    #         l = self.lam_upp
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

    def exact_line_search(self, x, d, l):
        lambda_d = - np.linalg.pinv(self.Q) @ self.q - x
        ind = np.where((lambda_d != 0) & (d != 0))
        lam = d[ind[0][0]] / lambda_d[ind[0][0]]
        # print("EXACT LAMBDA ", lam)
        return max(self.lam_low, min(self.lam_upp, abs(lam)))

    def _project(self, x):

        solver = dai_fletch_a1(self.left_constr, self.right_constr,
                               self.y, self.b, np.identity(self.n), x)
        xp = solver.solve(lam_i=1, d_lam=2, eps=self.projection_tol)

        # solver.plot_xtory()
        return xp

    def solve(self, x0, max_iter=10, tol=1e-3, x_opt=None, f_star=None):
        x = x0
        k = 0
        lam = 1

        if self.verbose:
            print(x)

        gradient = self.df(x)

        a = abs(1 / np.max(self._project(x - gradient) - x))
        gs = []
        ds = []
        fs = []
        fxs = []
        gap = []
        it_mu_rate = []
        rate = []
        rate_norm = []
        mu_rate = []
        xs = []
        orders = []
        time_proj = 0.0
        time_search = 0.0
        fxs.append(self.f(x))

        while k == 0 or k < max_iter:

            # print("before\t gradient={}\tx={}\ta={}".format(norm(gradient), norm(x), a))
            # print("K={}\tx={}".format(k, norm(x)))
            start_time_proj = time.time()
            d = self._project(x - a * gradient) - x

            time_proj += time.time() - start_time_proj
            # print("\t\tElapsed time in projection", time_proj)
            # print("projected d ={}".format(norm(d)))

            start_time_search = time.time()
            lam = self.line_search(x, d, lam)
            time_search += time.time() - start_time_search
            # print("lambda ", lam)

            x_prec = np.copy(x)
            fxs.append(self.f(x))

            x = x + lam * d
            gradient = self.df(x)
            if self.verbose:
                print("gradient {}\tx={}\td={}\tlambda={}".format(norm(gradient), norm(x), norm(d), lam))

            gs.append(norm(gradient))
            ds.append(norm(d))
            xs.append(norm(x))
            rate.append(norm(x - x_prec))
            rate_norm.append(rate[-1] / norm(x))

            if abs((fxs[-1] - fxs[-2])) / abs(fxs[-1]) < tol:
                print("Optimal solution found")
                break
            if rate_norm[-1] < tol:
                break
            if d.T @ self.Q @ d <= tol:
                if self.verbose:
                    print("amax")
                a = self.a_max
            else:
                a_new = self._select_updating_rule(d, a, lam)
                a = min(self.a_max, max(self.a_min, a_new))



            if f_star is not None:
                fs.append(abs((fxs[-1] - f_star) / f_star))
            if x_opt is not None:
                mu_rate.append(norm(x - x_opt) / norm(x_prec - x_opt))
            if k > 0:
                it_mu_rate.append(rate_norm[-1] / rate_norm[-2])
            # if k > 1:
            #     orders.append(np.log(rate_norm[-1]/rate_norm[-2])/np.log(rate_norm[-2]/rate_norm[-3]))

            k += 1

        if self.verbose:
            print("LAST K={}".format(k))
            print(norm(x), "  ", norm(gradient))
        if self.plots:
            self.plot_gradient([ds], ['proj gadient norm'], title="projected gradient norm", scale='log')
            self.plot_gradient([rate_norm], ['rate'], title="(x-x_prec)/norm(x)", scale='linear')
            # self.plot_gradient([rate], ['rate'], title="(x-x_prec)", scale='linear')
            # self.plot_gradient([orders], ['rate'], title="convergence order estimate = {}".format(np.mean(orders)), scale='linear')
            self.plot_gradient([it_mu_rate, mu_rate], ["empirical", "real"], title="mu", scale='log')
            self.plot_gradient([mu_rate], ["real"], title="convergence rate",
                               scale='linear', legend=False)
            if f_star is not None:
                self.plot_gradient([fs], ['gap'], title="f gap    f* = {}    f_opt = {}".format(f_star, fxs[-1]))
            # self.plot_gradient([xs], ['x norm'], title='x norm', scale='linear')
            # input()

        return x, d, time_proj, time_search

    def plot_gradient(self, histories, labels, title="", scale='log', legend=True):
        for i, h in enumerate(histories):
            plt.plot(range(0, len(h)), h, label=labels[i])
        plt.yscale(scale)
        plt.rcParams["figure.figsize"] = (10, 5)
        if legend:
            plt.legend()
        plt.title(title)
        plt.show()
