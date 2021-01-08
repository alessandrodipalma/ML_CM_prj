import numpy as np
from numpy import transpose as t


class LDBCQP:
    def __init__(self, q, Q, u, astart=1, m1=0.01, m2=0.9, eps=1e-6, max_feval=1000):
        self.q = q
        self.eps = eps
        self.Q = Q
        self.u = u
        self.n = len(q)
        self.feval = 0
        self.max_feval = max_feval
        self.dolh = False
        self.astart = astart
        self.m1 = m1
        self.m2 = m2
        self.lam = np.zeros((2 * self.n))
        self.d = np.zeros(self.n)

    def phild(self, alpha):
        """
        phi(lambda) is the lagrangian function of the problem
        phi(alpha) = phi(lambda + alpha d)
        t(phi(alpha)) =< lambda phi(lambda + alpha * d), d > 0
        :param a:
        :return:
        """
        p, self.lastg = self.phi(self.lam + alpha * self.d)
        pp = t(self.d) * self.lastg

        return p, pp

    def armijo_wolfe_ls(self, phi0, phip0, a_s, m1, m2, sfgrd=0.01, mina=1e-12):
        a = a_s
        phia, phips = self.phild(a)
        if np.any(phips <= 0):
            return a, phia
        else:
            lsiter = 1  # count ls iterations

            am = 0
            phipm = phip0

            while self.feval <= self.max_feval \
                    and (a_s - am) > mina \
                    and np.linalg.norm(phips) > 1e-12:
                # compute the new value by safeguarded quadratic interpolation
                a = (am * phips - a_s * phipm) / (phips - phipm)
                a = max([am * (1 + sfgrd) * min([a_s * (1 - sfgrd) * a])])

                # compute phi(a)
                phia, phip = self.phild(a)

                if phia <= phi0 + m1 * a * phip0 and abs(phip) <= -m2 * phip0:
                    break  # A + strong W is satisfied, we are done

                if phip <= 0:
                    am = a
                    phipm = phip
                else:
                    a_s = a
                    if a_s <= mina:
                        break
                    phips = phip

                lsiter += 1

        return a, phia

    def solve_lagrangian(self, lam):
        """The lagrangian relaxation of the problem is
        min(0.5*t(x)*Q*x + q*x - lambda^+ * (u-x) - lambda^- * x
        = min( 0.5*t(t)*Q*x + (q - lambda^+ - lambda^- ) * x - lambda^+ * u

        where lambda^+ are the first n components of lambda[]
              lambda^- are the last n components
              both constrained to be >=0

        The optimal solution of the Lagrangian relaxation is the unique solution of the linear system

            Q*x = -q - lambda^+ + lambda^-

        Since we have computed at the beginning the Cholesk0y factorization of Q,
        i.el Q = t(R) * R, R upper triangular, we obtain this by just 2 triangular backsolvers:

            t(R) * z = -q - lambda^+ + lambda^-

            R * x = z

        @:return the function value and the primal solution
        """
        q1 = self.q + lam[:self.n] - lam[self.n:]

        # R = np.linalg.cholesky(self.Q)
        # z = np.linalg.solve(t(R), t(-q1))
        y = np.linalg.solve(self.Q, q1)
        # compute phi
        p = (0.5 * t(y) * self.Q + t(q1)) * y - t(lam[:self.n]) * self.u

        self.feval += 1

        return p, np.ravel(y)

    def phi(self, lam):
        """
        This is the lagrangian function of the problem.
        With x the optimal solution of the minimization problem, the gradient at lambda is [x-u; -x]
        However, the line search is written for minimization but we rather want
        to maximize phi(), hence we have to change the sign of both function values and gradient entries.
        :param lam:
        :return:
        """
        p, y = self.solve_lagrangian(lam)
        p = -p
        g = np.append(self.u - y, y)

        if self.dolh:
            # compute an heuristic solution out of the solution y of the Lagrangian relaxation by projecting y on the
            # box
            y[y < 0] = 0
            indices = np.where(y > self.u)
            y[indices] = self.u[indices]

            # compute cost of feasible solution
            pv = 0.5 * t(y) * self.Q * y + t(self.q) * y

            if pv < self.v:  # it is better than the best one found so far
                self.x = y
                self.v = pv

        return p, g

    def solve_quadratic(self):
        # print("Solving problem with q={}, Q={}, u={}".format(self.q, self.Q, self.u))
        self.x = self.u / 2
        self.v = 0.5 * t(self.x) * self.Q + self.q * self.x

        p, self.lastg = self.phi(self.lam)

        while True:
            self.d = -self.lastg
            self.d = np.where(np.logical_and(self.lam <= 1e-12, self.d < 0), 0, self.d)

            if self.dolh:  # compute relative gap
                gap = (v + p) / max(abs(v))

                if gap <= self.eps:
                    print("OPT\n")
                    status = 'optimal'
                    break

            else:  # compute the norm of the projected gradient
                gnorm = np.linalg.norm(self.d)

                # print("{}\t{}\t{}".format(self.feval, -p, gnorm))

                if self.feval == 1:
                    gnorm0 = gnorm

                if gnorm <= self.eps * gnorm0:
                    print("OPT\n")
                    status = 'optimal'
                    break

            # stopping criteria

            if self.feval > self.max_feval:
                print("STOP\n")
                status = 'stopped'
                break

            # compute step size

            # first compute the maximum feasible step size maxt s.t. 0 <= lambda[i] + maxt * d[i]  for all i

            indices = self.d < 0
            if np.any(indices):

                min1 = min(-self.lam[indices] / d[indices])
                maxt = min(self.astart, min1)
            else:
                maxt = self.astart

            phip0 = t(self.lastg) * self.d
            a, p = self.armijo_wolfe_ls(p, phip0, maxt, self.m1, self.m2)
            # print("\t{}\n".format(a))

        return self.x
