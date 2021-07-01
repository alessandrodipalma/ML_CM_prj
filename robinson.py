import numpy as np
from numpy.linalg import norm

class robinson:

    def __init__(self, l, u, a, b):
        self.n = l.shape[0]

        if self.n != u.shape[0]:
            raise ValueError("u and l should have same length")
        if self.n != a.shape[0]:
            raise ValueError("a and l should have same length")

        self.l = l
        self.u = u
        self.a = a
        self.b = b

    def solve(self, x):
        if self.n != x.shape[0]:
            raise ValueError("x and l,u,a should have same length")
        I = set([i for i in range(x.shape[0])])
        L = set()
        U = set()
        R = np.copy(self.b)
        N = self.a.T @ self.a
        xp = np.copy(x)
        # print("solving with l={}, u= {}".format(self.l, self.u))
        x_prev = np.zeros(x.shape)
        while norm(xp - x_prev) > 1e-4:
            # Project O onto H
            # TODO parallelize
            for i in I:

                x_prev = np.copy(xp)
                xp[i] = (R / N) * self.a[i]
            print("\tR={}\tN={}\nxp={}".format(R, N, xp))
            # force x onto B
            sa = 0
            sb = 0
            lp = set()
            up = set()

            for i in I:
                if xp[i] < self.l[i]:
                    print("\tprojecting x[{}] on lower bound".format(i))
                    sb = sb + self.a[i] * (self.l[i] - xp[i])
                    lp.add(i)
                elif xp[i] > self.u[i]:
                    print("\tprojecting x[{}] on upper bound".format(i))
                    sa = sa + self.a[i] * (xp[i] - self.u[i])
                    up.add(i)

            if len(lp.union(up)) > 0:
                if sb > sa:
                    for i in lp:
                        N -= self.a[i] ** 2
                    for i in lp:
                        R -= self.a[i] * self.l[i]

                    I = I - lp
                    L = L.union(lp)
                elif sa > sb:
                    for i in up:
                        N -= self.a[i] ** 2
                    for i in up:
                        R -= self.a[i] * self.u[i]

                    I = I - up
                    U = U.union(up)
                else:
                    L = L + lp
                    U = U + up
            else:
                break

        x_opt = np.copy(xp)
        for i in L:
            x_opt[i] = self.l[i]
        for i in U:
            x_opt[i] = self.u[i]

        return x_opt
