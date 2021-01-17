import numpy as np
from gradientprojection import GradientProjection
from ldbcqp import LDBCQP
from matplotlib import pyplot as plt


def create_rbf_kernel(sigma):
    return lambda x, xi: np.exp(-np.inner(x - xi, x - xi) / (2 * sigma ** 2))


def create_poly_kernel(p):
    return lambda x, xi: (np.inner(x, xi) + 1) ** p


class SVM:
    # TODO labels with 0 value, or 0 data results in singular contraints matrix, should be fixed
    def __init__(self, kernel='rbf', C=0.0):
        """

        :type C: float
        """
        self.kernel_names = {'rbf': 'rbf', 'poly': 'poly'}
        self.kernel = self.__select_kernel(kernel)
        self.C = C

    def __select_kernel(self, kernel, sigma=1, p=1):
        if kernel == self.kernel_names['rbf']:
            return create_rbf_kernel(sigma)
        elif kernel == self.kernel_names['poly']:
            return create_poly_kernel(p)
        else:
            print("Not valid kernel name. Valid names are {}".format(self.kernel_names.values()))

    def compute_K(self, x):
        n = len(x)
        self.K = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                self.K[i][j] = self.kernel(x[i], x[j])
        return self.K

    def train(self, x, d, C=None):
        # print("training with x={}, d={}".format(x,d))
        if C is None:
            C = self.C
        if len(x) == len(d):
            n = len(x)
        else:
            print("X and y must have same size! Got X:{}, y:{}".format(x.shape, d.shape))
            pass

        K = self.compute_K(x)
        Q = np.empty(K.shape)

        for i in range(n):
            for j in range(n):
                Q[i, j] = d[i] * d[j] * K[i, j]

        # eig_Q, v = np.linalg.eig(Q)
        # eig_K, v = np.linalg.eig(K)
        # print("K Lmax/lmin=", np.max(eig_K) / np.min(eig_K))
        # print("Q Lmax/lmin=", np.max(eig_Q)/np.min(eig_Q), np.max(eig_Q), np.min(eig_Q))
        # print(K, Q)
        E = np.array([d])
        q = np.ones(n)




        alpha = GradientProjection(q=q, Q=Q, u=np.full(len(x), C),
                                   E=E, e=np.zeros(1)).solve()
        print(np.linalg.norm(alpha))
        # alpha = LDBCQP(q=np.ones(n), Q=Q, u=np.full(len(x), self.C)).solve_quadratic()
        # print("my = {}, frang = {}".format(alpha1, alpha))
        b = 0
        indexes = np.where(alpha > 1e-15)
        for j in indexes:
            sum = 0
            for i in range(n):
                sum += alpha[i] * d[i] * K[i, j]
            b += d[j] - sum

        self.b = b / len(indexes)
        self.alpha = alpha[indexes]
        self.d = d[indexes]
        self.x = x[indexes]

        return len(self.alpha), self.alpha

    def compute_out(self, x):
        f = lambda i: self.alpha[i] * self.d[i] * self.kernel(x, self.x[i])
        out = np.sum(np.array(list(map(f, np.arange(len(self.alpha))))))
        # print(out)
        return out

    def predict(self, x):
        out = np.array(list(map(self.compute_out, x)))
        # print(out)
        return out
