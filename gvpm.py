import time

import cvxpy
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm, matrix_power

from line_search import backtracking_armijo_ls
from knapsack_secant import dai_fletch_a1
from solver import Solver


class GVPM(Solver):
    """
    Solves a quadratic problem with box constraints using the Generalized Variable Projection Method.
    """

    STEPSIZE_BB = 'bb'

    class LineSearches:
        EXACT = 'exact'
        BACKTRACK = 'backtracking'

        values = [EXACT, BACKTRACK]

    class Plots:
        GAP = 'gap'
        X_NORM = 'x_norm'
        F_NORM = 'f_norm'
        F_RATE = 'f_rate'
        G_NORM = 'g_norm'
        D_NORM = 'd_norm'
        EXACT_RATE = 'ex_rate'
        ESTIMATE_RATE = 'est_rate'

    def __init__(self, ls=LineSearches.EXACT, a_min=1e-5, a_max=1e5, n_min=3, lam_low=1e-3, lam_upp=1, max_iter=100,
                 tol=1e-3,
                 verbose=False, proj_tol=1e-8, plots=True):
        """

        :param a_min:
        :param a_max:
        :param n_min:
        :param lam_low:
        :param lam_upp:


        """

        super().__init__(max_iter, tol, verbose)
        self.a_min = a_min
        self.a_max = a_max
        self.n_min = n_min
        self.n_max = 10
        self.current_rule = 1
        self.lam_low = lam_low
        self.lam_upp = lam_upp
        self.rule_iter = 1

        self.ls = ls
        self.projection_tol = proj_tol
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
        else:
            self.rule_iter += 1

        return a_new[self.current_rule]

    def _is_bad_descent_generator(self, a, a1, lam):
        return (lam < self.lam_low and a == a1) or (lam > self.lam_upp and a == a1)

    def _is_steplength_separator(self, a, a_new):
        return a_new[2] < a < a_new[1]

    def line_search(self, x, d, l):
        k = 0
        if self.ls == self.LineSearches.EXACT:
            l_new = abs((d.T @ d) / (d.T @ self.Q @ d))
        elif self.ls == self.LineSearches.BACKTRACK:
            l_new, it = backtracking_armijo_ls(self.f, self.df, x, d, alpha_min=self.lam_low)
            k += it
            # print("lambda_opt={}".format(l_new))

        if l_new is not None:
            l = l_new
        if l_new < self.lam_low:
            l = self.lam_low
        if l_new > self.lam_upp:
            l = self.lam_upp

        return l, k

    def _project(self, x, solver='dai_fletch'):
        if solver == 'dai_fletch':
            solver = dai_fletch_a1(self.left_constr, self.right_constr,
                                   self.y, self.b, np.identity(self.n), x)
            xp = solver.solve(lam_i=1, d_lam=2, eps=self.projection_tol)
            # solver.plot_xtory()
            return xp
        else:
            x_p = cvxpy.Variable(self.n)
            objective = cvxpy.Minimize((1 / 2) * cvxpy.quad_form(x_p, np.identity(self.n)) + x.T @ x_p)
            constraints = [x_p >= self.left_constr[0], x_p <= self.right_constr[0], self.y.T @ x_p == 0]
            problem = cvxpy.Problem(objective, constraints)
            problem.solve(verbose=True, solver="CPLEX", cplex_params={
                "barrier.convergetol": self.projection_tol
            })
            return np.array(x_p.value)

    def solve(self, x0, x_opt=None, f_opt=None):
        super().solve(x0, x_opt, f_opt)
        x = x0
        k = 0
        lam = 1
        start_time = time.time()
        eig_max_Q = np.linalg.eig(self.Q)[0][0]

        if self.verbose:
            print(x)

        gradient = self.df(x)

        a = abs(1 / np.max(self._project(x - gradient) - x))
        print(a)

        xs = [x]
        rate_norm = []
        rate = []
        fxs = []
        f_gaps = []
        if self.plots:
            gs = []
            ds = []

            it_mu_rate = []
            mu_rate = []
            f_rate = []
            upper_bounds_gap = []
            a_s = []
            fxs.append(self.f(x))

        time_proj = 0.0
        time_search = 0.0

        # convergence rate constants
        # tau = 1e-30
        # alpha_1 = self.lam_low / (2 * self.a_max)
        # alpha_2 = tau * (2 + 2 / self.a_min)
        # alpha_3 = (eig_max_Q.real * (alpha_2 + 1) + (2 / self.a_min)) * (alpha_2 + 1)
        # f_rate_estimate = alpha_3 / (alpha_3 + alpha_1)
        # input("alphas {} {} {} {} {} ".format(alpha_1, alpha_2, alpha_3, f_rate_estimate, eig_max_Q))

        it_ls = 0

        while k < self.max_iter:

            # print("before\t gradient={}\tx={}\ta={}".format(norm(gradient), norm(x), a))
            # print("K={}\tx={}".format(k, norm(x)))
            start_time_proj = time.time()

            d = self._project(x - a * gradient) - x

            time_proj += time.time() - start_time_proj
            # print("\t\tElapsed time in projection", time_proj)
            # print("projected d ={}".format(norm(d)))

            start_time_search = time.time()
            lam, it = self.line_search(x, d, lam)
            time_search += time.time() - start_time_search
            it_ls += it
            # print("lambda ", lam)

            x_prec = np.copy(x)

            x = x + lam * d
            gradient = self.df(x)
            if self.verbose:
                print("gradient {}\tx={}\td={}\tlambda={}".format(norm(gradient), norm(x), norm(d), lam))

            xs.append(norm(x))
            rate.append(norm(xs[-1] - xs[-2]))
            rate_norm.append(rate[-1] / xs[-1])
            fxs.append(self.f(x))

            if f_opt is not None:
                f_gaps.append(abs((fxs[-1] - f_opt) / f_opt))
                # if k > 0:
                #     f_rate.append(f_gaps[-1] / f_gaps[-2])

            if self.plots:
                gs.append(norm(gradient))
                ds.append(norm(d))
                a_s.append(a)

                if x_opt is not None:
                    upper_bounds_gap.append(
                        (eig_max_Q * (norm(x - x_opt) - ds[-1]) + (2 / self.a_min) * ds[-1]) * (
                                norm(x - x_opt) + ds[-1]))



                if x_opt is not None:
                    mu_rate.append(norm(x - x_opt) / norm(x_prec - x_opt))
                if k > 0:
                    it_mu_rate.append(rate_norm[-1] / rate_norm[-2])

            # Stopping criterions
            if k > 1 and rate_norm[-1] < self.tol:
                if self.verbose:
                    print("Optimal solution found")
                break

            # Update stepsize
            if d.T @ self.Q @ d <= self.tol:
                if self.verbose:
                    print("amax")
                a = self.a_max
            else:
                a_new = self._select_updating_rule(d, a, lam)
                a = min(self.a_max, max(self.a_min, a_new))

            k += 1
        end_time = time.time() - start_time

        if self.verbose:
            print("LAST K={}".format(k))
            print(norm(x), "  ", norm(gradient))
        if self.plots:
            # self.plot_gradient([ds], ['proj gadient norm'], title="projected gradient norm", scale='log')
            # self.plot_gradient([rate_norm], ['rate'], title="(x-x_prec)/norm(x)", scale='linear')
            # self.plot_gradient([rate], ['rate'], title="(x-x_prec)", scale='linear')
            # self.plot_gradient([orders], ['rate'], title="convergence order estimate = {}".format(np.mean(orders)), scale='linear')
            # self.plot_gradient([it_mu_rate, mu_rate], ["empirical", "real"], title="mu", scale='log')
            # self.plot_gradient([mu_rate], ["real"], title="convergence rate",
            #                    scale='linear', legend=False)
            if f_opt is not None:
                # self.plot_gradient([f_rate, [f_rate_estimate] * len(f_rate)], ['f_rate', 'f_rate_estimate'],
                #                    title="f gap    f* = {}    f_opt = {}".format(f_opt, fxs[-1]), scale='linear')
                self.plot_gradient([f_gaps], ["f_gap"])
            # self.plot_gradient([xs], ['x norm'], title='x norm', scale='linear')

        self.stats = {'it': k, 'it_ls': it_ls, 'time_tot': end_time, 'time_ls': time_search, 'time_proj': time_proj}
        self.f_gap_history = f_gaps
        self.rate_history = rate_norm
        self.x_history = xs

        return x, fxs[-1], d

    def plot_gradient(self, histories, labels, title="", scale='log', legend=True):
        for i, h in enumerate(histories):
            plt.plot(range(0, len(h)), h, label=labels[i])
        plt.yscale(scale)
        plt.rcParams["figure.figsize"] = (10, 5)
        if legend:
            plt.legend()
        plt.title(title)
        plt.show()
