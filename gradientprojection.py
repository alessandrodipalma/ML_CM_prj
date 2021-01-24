import numpy as np
from numpy import transpose as t
from numpy.linalg import inv as inv
from scipy.optimize import line_search
from matplotlib import pyplot as plt


def backtracking_armijo_ls(phi, d_phi, alpha, m1=0.9, tau=0.5):
    phi0 = phi(0)
    d_phi0 = d_phi(0)
    print("phi(alpha)={}\n(phi0 + m1 * alpha * d_phi0)={}".format(phi(alpha), (phi0 + m1 * alpha * d_phi0)))
    while phi(alpha) > (phi0 + m1 * alpha * d_phi0):
        alpha = tau * alpha

    return alpha


def armijo_wolfe_ls(phi: callable, d_phi: callable, a_max, m1=0.01, m2=0.9, eps=1e-6, max_iter=100, tau=0.5):
    phi_0 = phi_prev = phi(0)
    d_phi_0 = d_phi_prev = d_phi(0)
    a_prev = 0

    def interpolate(a_lo, a_hi, phi_lo, phi_hi, d_phi_lo, d_phi_hi):

        d1 = d_phi_lo + d_phi_hi - 3 * ((phi_lo - phi_hi) / (a_lo - a_hi))
        d2 = np.sign(a_hi - a_lo) * np.sqrt(d1 ** 2 - d_phi_lo * d_phi_hi)

        a = a_hi - (a_hi - a_lo) * (d_phi_hi + d2 - d1) / (d_phi_hi - d_phi_lo + 2 * d2)

        return a

    def zoom(a_lo, a_hi, phi_lo, phi_hi, d_phi_lo, d_phi_hi):

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
    if a_max is None:  # in our case, None == inf
        print("no max to a")
        a = 0.1
    else:
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

    def __init__(self, Q=None, q=None, u=None, x0=None, f=None, df=None, A=None, b=None, E=None, e=None, max_iter=100):
        """
        If the problem is quadratic, can specify it by Q, q, u s.t.: f(x) = 0.5* t(x) @ Q @ x + q @ x
        Otherwise, specify f(x) and its gradient df(x), A and b which defines the inequality constraints,
        and (E, e) which specify equality constraints.
        :param Q:
        :param q:
        :param u:
        :param x0:
        :param f:
        :param df:
        :param A:
        :param b:
        :param E:
        :param e:
        :param max_iter:
        """
        self.max_iter = max_iter
        if Q is None:
            if f is None:
                raise ValueError("f is None")
            else:
                if A is not None:
                    self.A = A
                    if b is not None:
                        self.b = b
                    else:
                        raise ValueError("b must be specified along with A")

                if E is not None:
                    self.E = E
                    if e is not None:
                        self.e = e
                    else:
                        raise ValueError("e must be specified along with E")

                if E is None and A is not None:
                    self.E = np.zeros((0, A.shape[1]))
                    self.e = np.zeros(0)
                elif A is None and E is not None:
                    self.A = np.zeros((0, E.shape[1]))
                    self.b = np.zeros(0)
                else:
                    raise ValueError("No constraints are defined. Use another algorithm for this problem!")

                self.f = f
                self.df = df

        elif Q.shape[0] == Q.shape[1]:
            if q is None:
                raise ValueError("q is None")
            elif q.shape[0] != Q.shape[0]:
                raise ValueError("Incompatible sizes: q has shape {}, but Q has shape {}".format(q.shape, Q.shape))

            self.f = lambda x: 0.5 * t(x) @ Q @ x + t(q) @ x
            self.df = lambda x: Q @ x

            if A is None:
                self.A = np.identity(Q.shape[0])
                self.b = u
            else:
                self.A = A

                if b is None:
                    raise ValueError("b is None but A has shape {}".format(A.shape))
                elif A.shape[0] == b.shape[0]:
                    self.b = b
                else:
                    raise ValueError("Incompatible sizes: b has shape {}, but A has shape {}".format(b.shape, A.shape))

            if E is None:
                self.E = np.zeros((0, Q.shape[1]))
                self.e = np.zeros(0)
            else:
                if e is not None:
                    self.e = e
                else:
                    raise ValueError("e must be specified along with E")
                self.E = E
        else:
            raise ValueError("Incompatible sizes Q={}, q={}, u={}".format(Q.shape, q.shape, u.shape))

        # Remove redundant equality constraints
        if self.E.shape[0] != 0:
            # print("self.E before", self.E)
            self.E = np.array([self.E[i] for i in range(self.E.shape[0]) if not np.array_equal(self.A[i], self.E[i])])
            # print("self.E after", self.E)

        self.x0 = x0

    def update_active_constraints(self, x, A, b, E, e, eps=1e-6):
        print(A, x)

        active_ineq_constr = np.where(
            np.logical_and(b - np.full(b.shape, eps) < A @ x, A @ x < b + np.full(b.shape, eps)))
        # print("A={}, x={}, A@x={}, b={}, A@x==b = {}".format(A,x,A@x,b,A@x==b))
        inactive_ineq_constr = np.where((A @ x) < (b - np.full(b.shape, eps)))
        A1 = A[active_ineq_constr]
        # print(active_ineq_constr)
        b1 = b[active_ineq_constr]
        A2 = A[inactive_ineq_constr]
        b2 = b[inactive_ineq_constr]

        active_eq_constr = np.where(E @ x == e)

        Q = E[active_eq_constr]

        return A1, A2, b1, b2, Q, active_ineq_constr[0]

    def step_2(self, d, b2, x, A2):
        # print("x= {}\nd={}".format(x, d))
        # print("b2={}\nA2@x={}".format(b2, A2 @ x))
        b_hat = b2 - A2 @ x
        d_hat = A2 @ d
        # print("A2={}, d={}".format(A2, d))
        if np.any(d_hat > 0):
            # print("b_hat={}\nd_hat={}\nb_hat/d_hat={}".format(b_hat, d_hat, (b_hat[d_hat > 0] / d_hat[d_hat > 0])))
            lambda_max = min((b_hat[d_hat > 0] / d_hat[d_hat > 0]))
        else:
            lambda_max = None
        # print("max step size = ", lambda_max)

        lambda_opt, fc, gc, new_fval, old_fval, new_slope = line_search(self.f, self.df, x, d, amax=lambda_max,
                                                                        maxiter=100, c1=0.01, c2=0.9)
        # lambda_opt = armijo_wolfe_ls(lambda a: self.f(x + a * d), lambda a: self.df(x + a * d) @ d, lambda_max)
        # lambda_opt = backtracking_armijo_ls(lambda a: self.f(x + a * d), lambda a:  self.df(x + a * d) @ d, lambda_max)

        # print("lambda_opt={}, x={}, d={}".format(lambda_opt, x, d))

        if lambda_opt is None:
            if lambda_max is not None:
                lambda_opt = lambda_max
            else:
                lambda_opt = 1

        # print("x = {} + {} * {}".format(x, lambda_opt, d))
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

        x_old = x - 1
        while k < self.max_iter and np.all(np.abs(x - x_old) > 1e-6):

            A1, A2, b1, b2, Q, I = self.update_active_constraints(x, self.A, self.b, self.E, self.e)

            # active constraints are the ones s.t. x[i] - u[i] == 0
            gradient = self.df(x)
            M = np.concatenate((A1, self.E))
            print("\n\nx{}={}\tI={}\tgradient={}".format(k, np.linalg.norm(x), I, np.linalg.norm(gradient)))

            if A1.shape[0] > 0:  # if there is any active constraint

                present_k = k

                while k == present_k:
                    # print(self.E.shape, self.E)

                    try:
                        P = np.identity(M.shape[1]) - M.T @ inv(M @ M.T) @ M
                    except np.linalg.LinAlgError:
                        print("M @ t(M) is singular with shape={}\n".format((M @ M.T).shape, axis=0), M)
                        raise np.linalg.LinAlgError

                    d = -P @ gradient
                    # print("P={}, g={}, d={}".format(P, gradient, d))

                    if np.all(np.equal(d, zero)) or np.all(np.abs(d) < 1e-6):

                        w = - inv(M @ M.T) @ M @ gradient
                        # print("w={}".format(w))
                        neg = np.where(w[:A1.shape[0]] < 0)[0]
                        if np.any(neg):
                            # print("negative components:{}".format(neg))
                            # remove the row corresponding to the first negative component from A1
                            A1 = np.delete(A1, neg[-1], 0)
                            b1 = np.delete(b1, neg[-1], 0)

                        else:
                            # x is a kkt point
                            print("x is KKT")
                            return x

                    else:
                        # STEP 2
                        # print("Going to line search")
                        x = self.step_2(d, b2, x, A2)
                        k += 1

            elif np.all(np.equal(gradient, zero)) or np.all(np.abs(gradient) < 1e-6):
                # stop, we're in a local minima
                print("gradient is zero, stopping")
                return x
            else:
                # we're inside the box, so we can act as if there is no constraint (almost)
                d = -gradient
                # STEP 2
                x = self.step_2(d, b2, x, A2)
                k += 1

        return x
