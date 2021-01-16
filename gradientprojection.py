import numpy as np
from numpy import transpose as t
from numpy.linalg import inv as inv
from scipy.optimize import line_search

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

    def __init__(self, Q=None, q=None, u=None, x0=None, f = None, df = None, A=None, b=None, max_iter=100):
        self.max_iter = max_iter
        if Q is None:
            if f is None:
                raise ValueError("f is None")
            else:
                self.f = f
                self.df = df
                self.A = A
                self.b = b
        elif Q.shape[0] == Q.shape[1] == q.shape[0] == u.shape[0]:
            self.f = lambda x: 0.5 * t(x) @ Q @ x + q @ x
            self.df = lambda x: Q @ x
            self.A = np.identity(Q.shape[0])
            self.b = u

        else:
            raise ValueError("Incompatible sizes Q={}, q={}, u={}".format(Q.shape, q.shape, u.shape))

        self.x0 = x0

    def update_active_constraints(self, x, A, b):

        active_constr = np.where(A @ x == b)
        # print("A={}, x={}, A@x={}, b={}, A@x==b = {}".format(A,x,A@x,b,A@x==b))
        inactive_constr = np.where(A @ x < b)
        A1 = A[active_constr]
        # print(active_constr)
        b1 = b[active_constr]
        A2 = A[inactive_constr]
        b2 = b[inactive_constr]
        return A1, A2, b1, b2, active_constr[0]

    def step_2(self, d, b2, x, A2):
        # print("x= {}\nd={}".format(x,d))
        print("b2={}\nA2={}".format(b2,A2))
        b_hat = b2 - A2 @ x
        d_hat = A2 @ d
        # print("A2={}, d={}".format(A2, d))
        # print("b_hat: {}, d_hat={}, d={}".format(b_hat.shape, d_hat.shape, d.shape))
        if np.any(d_hat > 0):
            print("b_hat={}\nd_hat={}".format(b_hat, d_hat))
            lambda_max = min((b_hat[d_hat > 0] / d_hat[d_hat > 0]))
        else:
            lambda_max = None
        print("max step size = ", lambda_max)
        # lambda_opt = armijo_wolfe_ls(lambda a: self.f(x + a * d), lambda a: self.df(x + a * d),
        #                              lambda_max)

        lambda_opt, fc, gc, new_fval, old_fval, new_slope = line_search(self.f, self.df, x, d, amax=lambda_max)
        print("optimum step size = ", lambda_opt)

        print("lambda_opt={}, x={}, d={}".format(lambda_opt, x, d))

        if lambda_opt is None:
            if lambda_max is not None:
                lambda_opt = lambda_max
            else:
                lambda_opt = 1

        print("x = {} + {} * {}".format(x, lambda_opt, d))
        x = x + lambda_opt * d

        return x

    def solve(self):
        u = self.b


        if self.x0 is None:
            x = u / 2
        else:
            x = self.x0


        zero = np.zeros(len(x))
        k = 1
        n = x.shape[0]
        while k<self.max_iter:


            A1, A2, b1, b2, I = self.update_active_constraints(x, self.A, self.b)
            print("\n\nx{}={}\tI={}".format(k, x, I))
            # active constraints are the ones s.t. x[i] - u[i] == 0
            gradient = self.df(x)

            if A1.shape[1] > 0:
                if A1.shape[1] == len(x):
                    print("all constraints are binding")
                present_k = k

                while k == present_k:
                    M = A1
                    # useless in case of M being all ones
                    # print(M.shape)
                    P = np.identity(M.shape[1]) - t(M) @ inv(M @ t(M)) @ M
                    d = -P @ gradient
                    print("P={}, g={}, d={}".format(P, gradient, d))
                    if np.all(np.equal(d, zero)):

                        w = - inv(M @ t(M)) @ M @ gradient
                        print("w={}".format(w))
                        neg = np.where(w < 0)[0]
                        if np.any(neg):
                            print("negative components:{}".format(neg))
                            # remove the row corresponding to the first negative component from A1
                            A1 = np.delete(A1, neg[-1], 0)
                            b1 = np.delete(b1, neg[-1], 0)

                        else:
                            # x is a kkt point
                            return x

                    else:
                        # STEP 2
                        print("Going to line search")
                        x = self.step_2(d, b2, x, A2)
                        k += 1

            elif np.all(np.equal(gradient, zero)):
                return x
            else:
                # print("here")
                d = -gradient
                # STEP 2
                x = self.step_2(d, b2, x, A2)
                k += 1

        return x
