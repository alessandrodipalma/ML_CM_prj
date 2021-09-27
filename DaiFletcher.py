import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm


class DaiFletcher:

    def __init__(self, l, u, a, b, A, c, verbose=False):
        """
        n = len(x)
        :param l: lower bounds
        :param u: upper bounds
        :param a: constraints s.t. a.T @ x = b
        :param b: scalar s.t. a.T @ x = b
        :param A: nxn matrix s.t. f(x) = 1/2 * x.T @ A @ x - c.T @ x
        :param c: n-sized vector s.t. f(x) = 1/2 * x.T @ A @ x - c.T @ x
        """
        self.l = l
        self.u = u
        self.a = a
        self.b = b
        self.A = A
        self.d = np.diagonal(A)
        self.c = c
        self.xtory = []
        self.verbose = verbose

        if self.verbose: print("solving knapsack with a = {} \n b = {} \n d = {}".format(self.a, self.b, self.d))

    def _bracketing(self, lam=0, d_lam=2, eps=1e-6):
        r = self.compute_r(lam)
        if self.verbose:
            print("R = {}".format(r))
        if r < 0:
            lam_l = lam
            r_l = r
            lam = lam + d_lam

            r = self.compute_r(lam)

            while r < -eps:
                lam_l = lam
                r_l = r

                s = max(r_l / r - 1, 0.1)
                d_lam = d_lam + d_lam / s
                lam = lam + d_lam

                if self.verbose:
                    print("R = {}".format(r))
                r = self.compute_r(lam)

            lam_u = lam
            r_u = r
        else:
            lam_u = lam
            r_u = r
            lam = lam - d_lam

            r = self.compute_r(lam)

            while r > eps:
                lam_u = lam
                r_u = r
                s = max(r_u / r - 1, 0.1)

                d_lam = d_lam + d_lam / s
                lam = lam - d_lam

                if self.verbose:
                    print("R = {}".format(r))

                r = self.compute_r(lam)

            lam_l = lam
            r_l = r

        return d_lam, lam_l, lam_u, r_l, r_u

    def _secant(self, lam_i=0, d_lam=2, eps=1e-2):
        d_lam, lam_l, lam_u, r_l, r_u = self._bracketing(lam_i, d_lam)

        if r_l == 0.0 or r_u == 0.0:  # it's a KT point
            return
        if self.verbose:
            print(d_lam, lam_l, lam_u, r_l, r_u)
        s = 1 - r_l / r_u
        d_lam = d_lam / s
        lam = lam_u - d_lam
        r = self.compute_r(lam)

        while abs(r) > eps:
            if r > eps:
                if s <= 2:
                    lam_u = lam
                    r_u = r
                    s = 1 - r_l / r_u
                    d_lam = (lam_u - lam_l) / s
                    lam = lam_u - d_lam
                else:
                    s = max(r_u / r - 1, 0.1)
                    d_lam = (lam_u - lam) / s
                    lam_new = max(lam - d_lam, 0.75 * lam_l + 0.25 * lam)
                    lam_u = lam
                    r_u = r
                    lam = lam_new
                    s = (lam_u - lam_l) / (lam_u - lam)
            else:
                if s >= 2:
                    lam_l = lam
                    r_l = r
                    s = 1 - r_l / r_u
                    d_lam = (lam_u - lam_l) / s
                    lam = lam_u - d_lam
                else:
                    s = max(r_l / r - 1, 0.1)
                    d_lam = (lam - lam_l) / s
                    lam_new = min(lam + d_lam, 0.75 * lam_u + 0.25 * lam)
                    lam_l = lam
                    r_l = r
                    lam = lam_new
                    s = (lam_u - lam_l) / (lam_u - lam)
            r = self.compute_r(lam)

        return

    def solve(self, lam_i=0, d_lam=2, eps=1e-2):
        self._secant(lam_i, d_lam, eps=eps)
        return self.x

    def compute_h(self, lam):
        lam = float(lam)
        # print("lamda = {}\n".format(lam))
        return (self.c + lam * self.a) / self.d

    def compute_x(self, lam):
        h = self.compute_h(lam)
        # print(np.array([self.l, h, self.u]).T, "\n")
        return np.median(np.array([self.l, h, self.u]).T, axis=1)

    def compute_r(self, lam):
        self.x = self.compute_x(lam)
        self.xtory.append(norm(self.x))
        return float(self.a @ self.x - self.b)

    def plot_xtory(self, title="xtory", color="g"):
        plt.plot(range(0, len(self.xtory)), self.xtory, c=color)
        plt.title(title)
        plt.yscale('log')
        plt.show()
