import numpy as np
from matplotlib import pyplot
from sklearn.datasets import make_classification

from gradientprojection import GradientProjection
from ldbcqp import LDBCQP

def create_rbf_kernel(sigma):
    return lambda x, xi: np.exp(-np.inner(x - xi, x - xi) / (2 * sigma ** 2))


def create_poly_kernel(p):
    return lambda x, xi: (np.inner(x, xi) + 1) ** p

class SVM:

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

    def train(self, x, d):
        # print("training with x={}, d={}".format(x,d))

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

        alpha = GradientProjection(q=np.ones(n), Q=Q, u=np.full(len(x), self.C)).solve()
        # alpha = LDBCQP(q=np.ones(n), Q=Q, u=np.full(len(x), self.C)).solve_quadratic()
        # print("my = {}, frang = {}".format(alpha1, alpha))
        b = 0
        indexes = np.where(alpha > 0)
        for j in indexes:
            sum = 0
            for i in range(n):

                sum += alpha[i] * d[i] * K[i, j]
            b += d[j] - sum

        self.b = b / len(indexes)
        self.alpha = alpha[indexes]
        self.d = d[indexes]
        self.x = x[indexes]

        return len(self.alpha)

    def compute_out(self, x):
        f = lambda i: self.alpha[i] * self.d[i] * self.kernel(x, self.x[i])
        out = np.sum(np.array(list(map(f, np.arange(len(self.alpha))))))
        # print(out)
        return out

    def predict(self, x):
        out = np.array(list(map(self.compute_out, x)))
        # print(out)
        return out








