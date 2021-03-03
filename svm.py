import numpy as np
from convergent_rosen import GradientProjection
from gvpm import GVPM
from ldbcqp import LDBCQP
from matplotlib import pyplot as plt
from cvxopt import solvers, matrix



def create_rbf_kernel(gamma):
    print("Rbf kernel with sigma = {}".format(gamma))

    return lambda x, y: np.exp(- gamma * np.square((np.linalg.norm(x - y))))


def create_poly_kernel(p):
    print("Polinomial kernel with grade {}".format(p))
    return lambda x, xi: (np.inner(x, xi) + 1) ** p


class SVM:
    KERNELS = {'rbf': 'rbf', 'poly': 'poly', 'linear': 'linear'}

    # TODO labels with 0 value, or 0 data results in singular contraints matrix, should be fixed
    def __init__(self, kernel='rbf', C=1.0, sigma=1, degree=3):
        """

        :type C: float
        """
        self.kernel_name = kernel
        self.kernel = self._select_kernel(kernel, sigma, degree)
        self.C = C

    def _select_kernel(self, kernel, sigma=1, p=3):
        if kernel == SVM.KERNELS['rbf']:
            return create_rbf_kernel(sigma)
        elif kernel == SVM.KERNELS['poly']:
            return create_poly_kernel(p)
        elif kernel == SVM.KERNELS['linear']:
            return create_poly_kernel(1)
        else:
            print("Not valid kernel name. Valid names are {}".format(SVM.KERNELS.values()))

    # def compute_gaussian_k(self, x, gamma):
    #

    def compute_K(self, x):
        n = len(x)
        self.K = np.empty((n, n))
        for i in range(n):
            for j in range(i+1):
                ij = self.kernel(x[i], x[j])
                self.K[j][i] = ij
                self.K[i][j] = ij

        return self.K



    def train(self, x, d, C=None, sigma=1):
        self.kernel = self._select_kernel(self.kernel_name, sigma=sigma)

        # print("training with x={}, d={}".format(x,d))
        if C is None:
            C = self.C
        if len(x) == len(d):
            n = len(x)
        else:
            print("X and y must have same size! Got X:{}, y:{}".format(x.shape, d.shape))
            pass

        K = self.compute_K(x)
        # print("Kernel:", K)
        Q = np.empty(K.shape)

        for i in range(n):
            for j in range(n):
                Q[i, j] = d[i] * d[j] * K[i, j]

        alpha = self.solve_optimization(C, d, n, Q)
        print(alpha)
        # alpha = LDBCQP(q=np.ones(n), Q=Q, u=np.full(len(x), self.C)).solve_quadratic()
        # print("my = {}, frang = {}".format(alpha1, alpha))
        b = 0
        indexes = np.where(alpha > (C / 10000))[0]
        # print(alpha)
        for j in indexes:
            sum = 0
            for i in range(n):
                sum += alpha[i] * d[i] * K[i, j]
            b += d[j] - sum

        try:
            self.b = np.array(b / len(indexes))
        except ZeroDivisionError:
            self.b = []

        self.alpha = np.array(alpha[indexes])
        self.d = np.array(d[indexes])
        self.x = np.array(x[indexes])

        return len(self.alpha), self.alpha, indexes

    def solve_optimization(self, C, d, n, Q):
        """

        :param C: regularization parameter
        :param d: desired output
        :param n: input vector dimension
        :return:
        """
        q = - np.ones(n)
        A = np.append(np.identity(n), np.diag(np.full(n, -1)), axis=0)
        b = np.append(np.full(n, C), np.zeros(n))
        E = d
        e = np.zeros((1, 1))

        # alpha = GradientProjection(f=lambda x: 0.5 * x.T @ Q @ x + q @ x,
        #                            df=lambda x: Q @ x + q,
        #                            A=A, b=b, Q=E.reshape((1, E.shape[0])), q=e) \
        #     .solve(x0=np.full(n, 0))

        alpha = GVPM(f=lambda x: 0.5 * x.T @ Q @ x + q @ x,
                                   df=lambda x: Q @ x + q,
                                   A=A, b=b) \
            .solve(x0=np.full(n, C/2))
        return alpha

    def compute_out(self, x):
        f = lambda i: self.alpha[i] * self.d[i] * self.kernel(x, self.x[i])
        out = np.sum(np.array(list(map(f, np.arange(len(self.alpha))))))
        # print(out)
        return out

    def predict(self, x):
        # print("PREDICTING...\nalpha={}".format(self.alpha))
        out = np.array(list(map(self.compute_out, x)))
        # print(out)
        return np.sign(out)
