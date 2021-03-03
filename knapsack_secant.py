import numpy as np


class dai_fletch_a1:

    def __init__(self, l, u, a, b, A, c):
        self.l = l
        self.u = u
        self.a = a
        self.b = b
        self.A = A
        self.d = np.diagonal(A)
        self.c = c

        print("solving knapsack with a = {} \n b = {} \n d = {}".format(self.a, self.b, self.d))

    def _bracketing(self, lam=0, d_lam=2, max_iter):
        r = self.compute_r(lam)

        if r < 0:
            lam_l = lam
            r_l = r
            lam = lam + d_lam

            r = self.compute_r(lam)

            while r < 0 and k < max_iter:
                lam_l = lam
                r_l = r

                s = max(r_l / r - 1, 0.1)
                d_lam = d_lam + d_lam / s
                lam = lam + d_lam

                r = self.compute_r(lam)

            lam_u = lam
            r_u = r
        else:
            lam_u = lam
            r_u = r
            lam = lam - d_lam
            r = self.compute_r(lam)

            while r > 0 and k < max_iter:
                lam_u = lam
                r_u = r
                s = max(r_u / r - 1, 0.1)

                d_lam = d_lam + d_lam / s
                lam = lam - d_lam

                r = self.compute_r(lam)

            lam_l = lam
            r_l = r

        return d_lam, lam_l, lam_u, r_l, r_u

    def _secant(self, eps=1e-8):
        d_lam, lam_l, lam_u, r_l, r_u = self._bracketing()
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

    def solve(self):
        self._secant()
        return self.x

    def compute_h(self, lam):
        lam = np.float(lam)
        print("lamda = {}\n".format(lam))
        return (self.c + lam * self.a) / self.d

    def compute_x(self, lam):
        h = self.compute_h(lam)
        print(np.array([self.l, h, self.u]).T, "\n")
        return np.median(np.array([self.l, h, self.u]).T, axis=1)

    def compute_r(self, lam):
        self.x = self.compute_x(lam)
        return self.a @ self.x - self.b
