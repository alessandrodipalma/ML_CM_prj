import numpy as np
from numpy import transpose as t, invert as inv


def backtracking_armijo_ls(phi, d_phi, alpha, m1, tau):
    phi0 = phi(0)
    d_phi0 = d_phi(0)
    while phi(alpha) > (phi0 + m1 * alpha * d_phi0):
        alpha = tau * alpha

    return alpha


def armijo_wolfe_ls(phi: callable, d_phi: callable, a_max: float, m1=0.01, m2=0.9, eps=1e-6, max_iter=100, tau=0.5):
    phi_0 = phi_prev = phi(0)
    d_phi_0 = d_phi_prev = d_phi(0)
    a_prev = 0

    def interpolate(a_lo: float, a_hi: float, phi_lo: float, phi_hi: float, d_phi_lo: float, d_phi_hi: float):

        d1 = d_phi_lo + d_phi_hi - 3 * ((phi_lo - phi_hi) / (a_lo - a_hi))
        d2 = np.sign(a_hi - a_lo) * np.sqrt(d1 ** 2 - d_phi_lo * d_phi_hi)

        a = a_hi - (a_hi - a_lo) * (d_phi_hi + d2 - d1) / (d_phi_hi - d_phi_lo + 2 * d2)

        return a

    def zoom(a_lo: float, a_hi: float, phi_lo: float, phi_hi: float, d_phi_lo: float, d_phi_hi: float):

        while True:
            a_j = interpolate(a_lo, a_hi, phi_lo, phi_hi, d_phi_lo, d_phi_hi)

            phi_j = phi(a_j)
            d_phi_j = d_phi(a_j)

            if phi_j > phi_0 + m1 * a_j * d_phi_0 \
                    or phi_j >= phi_lo:
                a_hi = a_j
                d_phi_hi = d_phi_j
            else:
                if abs(d_phi_j) <= -m2 * d_phi_0:  # goldstein
                    return a_j
                if d_phi_j * (a_hi - a_lo) >= 0:  # it's increasing, shift the interval
                    a_hi = a_lo
                a_lo = a_j
                d_phi_lo = d_phi_j

    # if np.any(d_phi_x <= 0):
    #     return a_max
    a = a_max
    i = 1

    while i <= max_iter \
            and abs(a - a - a_prev) > eps \
            and np.all(phi_prev > 1e-12):
        phi_a = phi(a)
        d_phi_a = d_phi(a)

        if phi_a > phi_0 + m1 * a * d_phi_0 \
                or (phi_a >= phi_prev and i > 1):
            return zoom(a_prev, a, phi_prev, phi_a, d_phi_a, d_phi_prev)

        if abs(d_phi_a) <= -m2 * d_phi_0:
            return a
        if d_phi_a >= 0:
            return zoom(a, a_prev, phi_a, phi_prev, d_phi_a, d_phi_prev)
        a_prev = a
        d_phi_prev = d_phi_a
        phi_prev = phi_a
        a = a * tau
        # print(a)

    return a


class GradientProjection:

    def __init__(self, Q, q, u, f = None, df = None):
        if Q.shape[0] == Q.shape[1] == q.shape[0] == u.shape[0]:
            self.f = lambda x: 0.5 * t(x) @ Q @ x + q
            self.df = lambda x: Q @ x
            self.u = u
        else:
            raise ValueError("Incompatible sizes Q={}, q={}, u={}".format(Q.shape, q.shape, u.shape))

    def update_active_constraints(self, x, u):
        ide = np.identity(len(x))
        active_constr = np.where((x - u) == 0)
        inactive_constr = np.where((x - u) != 0)
        A1 = ide[active_constr]
        # print(active_constr)
        b1 = u[active_constr]
        A2 = ide[inactive_constr]
        b2 = u[inactive_constr]
        return A1, A2, b1, b2, np.array(active_constr)

    def step_2(self, d, b2, x, A2):
        # print("x= {}\nd={}".format(x,d))
        b_hat = b2 - A2 @ x
        d_hat = A2 @ d
        # print("b_hat: {}, d_hat={}, d={}".format(b_hat.shape, d_hat.shape, d.shape))
        lambda_max = min((b_hat / d_hat)[d_hat > 0])
        # print("max step size = ", lambda_max)
        lambda_opt = armijo_wolfe_ls(lambda a: self.f(x + a * d), lambda a: self.df(x + a * d),
                                     lambda_max)
        # print("optimum step size = ", lambda_opt)
        x = x + lambda_opt * d

        return x

    def solve(self):
        u = self.u

        x = u / 2
        zero = np.zeros(len(x))
        k = 1

        while k<100:

            print(k)
            A1, A2, b1, b2, I = self.update_active_constraints(x, u)

            # active constraints are the ones s.t. x[i] - u[i] == 0
            gradient = self.df(x)

            if I.shape[1] > 0:
                if I.shape[1] == len(x):
                    print("all constraints are binding")
                present_k = k

                while k == present_k:
                    M = A1

                    # useless in case of M being all ones
                    # print(M.shape)
                    if I.shape[1] == 1:
                        P = np.identity(M.shape[1])
                    else:
                        P = np.identity(M.shape[1]) - t(M) @ inv(M @ t(M)) @ M
                    d = -(P @ gradient)

                    if np.all(np.equal(d, zero)):
                        w = -inv(M @ t(M)) @ M @ gradient

                        neg = np.where(w < 0)
                        if any(neg):
                            # remove the row corresponding to the first negative component from A1
                            A1 = np.delete(A1, neg[0], 0)
                            b1 = np.delete(b1, neg[0], 0)

                        else:
                            # x is a kkt point
                            break

                    else:
                        # STEP 2
                        x = self.step_2(d, b2, x, A2)
                        k += 1

            elif np.all(np.equal(gradient, zero)):
                break
            else:
                d = -gradient
                # STEP 2
                x = self.step_2(d, b2, x, A2)
                k += 1

        return x
