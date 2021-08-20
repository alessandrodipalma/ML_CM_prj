import numpy as np
from sklearn import preprocessing

from gvpm import GVPM
from solver import Solver


class  SVM:
    KERNELS = {'rbf': 'rbf', 'poly': 'poly', 'linear': 'linear'}

    # TODO labels with 0 value, or 0 data results in singular contraints matrix, should be fixed
    def __init__(self, solver: Solver, exact_solver=None, kernel='rbf', C=1.0, gamma='scale', degree=3):
        """

        :type C: float
        """
        self.kernel_name = kernel
        self.kernel = self._select_kernel(kernel, gamma, degree)
        self.C = C
        self.gamma = gamma
        self.gamma_value = None
        self.solver = solver
        self.exact_solver = exact_solver
        self.verbose = False


    def create_rbf_kernel(self):
        # print("Rbf kernel with sigma = {}".format(gamma))

        return lambda x, y: np.exp(- self.gamma_value * np.square((np.linalg.norm(x - y))))

    def create_poly_kernel(self, p):
        print("Polinomial kernel with grade {}".format(p))
        return lambda x, xi: (self.gamma_value * np.inner(x, xi) + 1) ** p

    def _select_kernel(self, kernel, gamma, degree):
        if kernel == SVM.KERNELS['rbf']:
            return self.create_rbf_kernel()
        elif kernel == SVM.KERNELS['poly']:
            return self.create_poly_kernel(degree)
        elif kernel == SVM.KERNELS['linear']:
            return self.create_poly_kernel(1)
        else:
            print("Not valid kernel name. Valid names are {}".format(SVM.KERNELS.values()))

    # def compute_gaussian_k(self, x, gamma):
    #

    def compute_K(self, x):
        n = len(x)
        self.K = np.empty((n, n))
        for i in range(n):
            for j in range(i + 1):
                ij = self.kernel(x[i], x[j])
                self.K[j][i] = ij
                self.K[i][j] = ij

        return self.K

    def train(self, x, d):

        if len(x) == len(d):
            n = len(x)
        else:
            print("X and y must have same size! Got X:{}, y:{}".format(x.shape, d.shape))
            pass

        if self.gamma == 'auto':
            self.gamma_value = 1 / n
        elif self.gamma == 'scale':
            self.gamma_value = 1 / (n * x.var())

        K = self.compute_K(x)
        # print("Kernel:", K)
        Q = np.empty(K.shape)

        for i in range(n):
            for j in range(n):
                Q[i, j] = d[i] * d[j] * K[i, j]

        alpha = self.solve_optimization(d, Q)
        print(alpha)
        # alpha = LDBCQP(q=np.ones(n), Q=Q, u=np.full(len(x), self.C)).solve_quadratic()
        # print("my = {}, frang = {}".format(alpha1, alpha))
        b = 0
        indexes = np.where(alpha > 1e-6)[0]
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

    def solve_optimization(self, d, Q):
        """

        :param C: regularization parameter
        :param d: desired output
        :param n: input vector dimension
        :return:
        """
        n = Q.shape[0]
        q = - np.ones(n)
        l = np.full(n, 0.)
        u = np.full(n, float(self.C))

        self.solver.define_quad_objective(Q, q, l, u,d, 0)
        alpha = self.solver.solve(x0=np.full(n, self.C / 2))[0]
        return alpha

    def compute_out(self, x):
        f = lambda i: self.alpha[i] * self.d[i] * self.kernel(x, self.x[i])
        out = np.sum(np.array(list(map(f, np.arange(len(self.alpha)))))) + self.b
        # print(out)
        return out

    def parallel_predict(self, x):
        return np.array(list(map(self.compute_out, x)))

    def predict(self, x):
        # print("PREDICTING...\nalpha={}".format(self.alpha))
        out = self.parallel_predict(x)
        # print(out)
        return np.sign(out)
