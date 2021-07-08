from copy import deepcopy

import numpy as np
from numpy import array
from numpy.linalg import inv, norm, pinv
from scipy.optimize import line_search
from line_search import armijo_wolfe_ls


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
                raise ValueError("Incompatible sizes: Q and q must have the same amount of rows, but Q has shape "
                                 "{} and q has shape {}".format(Q.shape, q.shape))

        '''
        Even if equality or inequality constraints are not specified, a vacuous matrix and vector are
        created to avoid checks in the rest of the algorithm
        '''

        if Q is None and A is None:
            raise ValueError("No constraints specified")
        elif Q is None:
            self.n = A.shape[1]
            self.A = A
            self.b = b
            self.Q = np.zeros((0, self.n))
            self.q = np.array([])
        elif A is None: # A is None
            self.n = Q.shape[1]
            self.Q = Q
            self.q = q
            self.A = np.zeros((0, self.n))
            self.b = np.array([])
        else:
            self.n = Q.shape[1]
            self.A = A
            self.b = b
            self.Q = Q
            self.q = q


        self.I = np.identity(self.n)
        print("A has shape {}\n"
              "Q has shape {}".format(self.A.shape, self.Q.shape))



    def update_active_constraints(self, x, A, b, eps=1e-15):
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

        lambda_opt, fc, gc, new_fval, old_fval, new_slope = line_search(self.f, self.df, x, d, amax=lambda_max, c1=0.1, c2=0.9)
        # lambda_opt = armijo_wolfe_ls(self.f, self.df, x, d, lambda_max, max_iter=20, m1=0.1, m2=0.9)

        if lambda_opt is None:
            if lambda_max is not None:
                lambda_opt = lambda_max
            else:
                lambda_opt = 1

        x = x + lambda_opt * d

        return x

    def solve(self, x0,  maxiter = 100, c=1e-8, delta_d = 0.3):
        x = x0
        x_old = np.ones(self.n)
        k = 0
        I = np.identity(x.shape[0])
        d = np.zeros(self.n)
        d_old = np.ones(self.n)

        constraint_matrix = np.concatenate((self.A, self.Q))
        constraints_bounds = np.append(self.b, self.q)

        if self.Q @ x != self.q:
            print("x0 doesn't respects equality constraints defined by Q = {}, q ={}".format(self.Q, self.q))

        while k < maxiter and d is not None:

            A1, A2, b1, b2, active_constraints = self.update_active_constraints(x, self.A, self.b)
            M = np.concatenate((A1, self.Q), axis=0)
            # M = A1
            # M = M[~np.all(M==0, axis=1)]
            # print(M)
            gradient = self.df(x)

            if M.shape[0] == 0:
                print("M is vacuous")
                if np.all(gradient <= 1e-6):
                    print("no constraints, gradient is 0")
                    d = None
                else:
                    d_old = deepcopy(d)
                    d = -gradient
            else:
                # print("{} active constraints".format(active_constraints.shape[0]))

                d1 = self.project(M, gradient)
                w = - inv(M @ M.T) @ M @ gradient
                u = w[:A1.shape[0]]

                # print("P={}\n\nu={}".format(P, u))
                if np.all(u >= -1e-15):
                    if np.all(d == 0):
                        # print("kkt point")
                        d = None
                    else:
                        # print("d is d1")
                        d_old = deepcopy(d)
                        d = d1

                else:
                    uh = np.min(u)
                    Mh = M[np.where(u != uh)]

                    if np.linalg.norm(d1) > abs(uh) * c:
                        # print("d is d1")
                        d_old = deepcopy(d)
                        d = d1
                    else:
                        # print("d is d2")

                        d2 = self.project(Mh, gradient)
                        d_old = deepcopy(d)
                        d = d2

            if d is not None:
                x_old = deepcopy(x)

                # print("k={}, ||g|| = {},  ||d||= {}, x={}, d-d_old={}".format(k, norm(gradient), norm(d),norm(x), norm(d_old)-norm(d)))

                # if norm(d_old) - norm(d) < delta_d and k > 1:
                #     d = None
                # else:
                x = self.step_2(d, x, A2, b2)

                k += 1

        # print("converged in {} iterates. ||x|| = {}, ||g|| = {}".format(k, norm(x), norm(gradient)))

        return x

    def project(self, M, gradient):
        print("M\n", M)

        print("gradient\n", gradient)

        I = self.I
        print(M.shape)
        print(M @ M.T)
        P = I - M.T @ inv(M @ M.T) @ M

        d1 = np.linalg.solve(P, gradient)

        print("projected gradient\n", d1)
        return d1









