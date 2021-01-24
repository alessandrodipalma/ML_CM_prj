import numpy as np
from numpy import array
from numpy.linalg import inv, norm
from scipy.optimize import line_search


class GradientProjection:

    def __init__(self, f, df, A=None, b=None, Q=None, q=None):
        """

        :param f: objective function
        :param df: objective function gradient
        :param A: inequality constraints coefficient matrix
        :param b: inequality constraints bound vector
        :param Q: equality constraints coefficient matrix
        :param q: equality constraints bound vector
        """

        self.f = f
        self.df = df

        if A is None:
            if b is not None:
                raise ValueError("b has shape {} but A is None".format(b.shape))
        else:
            if b is None:
                raise ValueError("A has shape {} but b is None".format(A.shape))
            if A.shape[0] != b.shape[0]:
                raise ValueError("Incompatible sizes: A and b must have the same amount of rows, but A has shape "
                                 "{} and b has shape {}".format(A.shape, b.shape))
        if Q is None:
            if q is not None:
                raise ValueError("b has shape {} but A is None".format(b.shape))
        else:
            if q is None:
                raise ValueError("A has shape {} but b is None".format(Q.shape))
            if Q.shape[0] != q.shape[0]:
                raise ValueError("Incompatible sizes: A and b must have the same amount of rows, but A has shape "
                                 "{} and b has shape {}".format(Q.shape, q.shape))

        '''
        Even if equality or inequality constraints are not specified, a vacuous matrix and vector are
        created to avoid checks in the rest of the algorithm
        '''

        if Q is None and A is None:
            raise ValueError("No constraints specified")
        if Q is None:
            self.n = A.shape[1]
            self.Q = np.zeros((0, self.n))
            self.q = np.array([])
        if A is None: # A is None
            self.n = Q.shape[1]
            self.A = np.zeros((0, self.n))
            self.b = np.array([])
        else:
            self.n = Q.shape[1]
            self.A = A
            self.b = b
            self.Q = Q
            self.q = q



    def update_active_constraints(self, x, A, b, eps=1e-16):
        residual = A @ x - b
        active_ineq_constr = np.where(abs(residual) < eps)
        inactive_ineq_constr = np.where(abs(residual) > eps)
        # print("active:{}\ninactive:{}".format(active_ineq_constr, inactive_ineq_constr))
        A1 = A[active_ineq_constr]
        b1 = b[active_ineq_constr]
        A2 = A[inactive_ineq_constr]
        b2 = b[inactive_ineq_constr]

        return A1, A2, b1, b2, active_ineq_constr[0]

    def step_2(self, d, x, A2, b2):
        b_hat = b2 - A2 @ x
        d_hat = A2 @ d
        if np.any(d_hat > 0):
            lambda_max = min((b_hat[d_hat > 0] / d_hat[d_hat > 0]))
        else:
            lambda_max = None

        lambda_opt, fc, gc, new_fval, old_fval, new_slope = line_search(self.f, self.df, x, d, amax=lambda_max,
                                                                        maxiter=100, c1=0.1, c2=0.9)

        if lambda_opt is None:
            if lambda_max is not None:
                lambda_opt = lambda_max
            else:
                lambda_opt = 1

        x = x + lambda_opt * d

        return x

    def solve(self, x0, maxiter = 100, c=0.5):
        x = np.zeros(self.n)

        k = 0
        I = np.identity(x.shape[0])
        d = 0

        constraint_matrix = np.concatenate((self.A, self.Q))
        constraints_bounds = np.append(self.b, self.q)
        if self.Q @ x != self.q:
            print("x0 doesn't respects equality constraints defined by Q = {}, q ={}".format(self.Q, self.q))

        while k < maxiter and d is not None:

            A1, A2, b1, b2, active_constraints = self.update_active_constraints(x, self.A, self.b)
            # M = np.concatenate((A1, self.Q))
            M = A1
            gradient = self.df(x)
            # print("k={}, ||g|| = {},  ||d||= {}, x={}".format(k, norm(gradient), norm(d), norm(x)))
            if M.shape[0] == 0:

                if np.all(gradient <= 1e-6):
                    print("no constraints, gradient is 0")
                    d = None
                else:
                    d = -gradient
            else:
                # print("active constraints: ", active_constraints)

                P = I - M.T @ inv(M @ M.T) @ M
                d1 = - P @ gradient
                w = - inv(M @ M.T) @ M @ gradient
                u = w[:A1.shape[0]]

                if np.all(u >= -1e-15):
                    if np.all(d == 0):
                        print("kkt point")
                        d = None
                    else:
                        d = d1

                else:
                    uh = np.min(u)
                    A1h = A1[np.where(u != uh)]
                    Mh = np.concatenate((A1h, self.Q))
                    Ph = I - Mh.T @ inv(Mh @ Mh.T) @ Mh

                    if np.linalg.norm(d1) > abs(uh) * c:
                        d = d1
                    else:
                        d2 = - Ph @ gradient
                        d = d2

            if d is not None:
                x = self.step_2(d, x, A2, b2)
                k += 1

        print("converged in {} iterates. x = {}".format(k, x))

        return x









