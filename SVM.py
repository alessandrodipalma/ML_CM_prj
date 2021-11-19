import numpy as np
from Solver import Solver


class SVM:
    KERNELS = {'rbf': 'rbf', 'poly': 'poly', 'linear': 'linear'}

    def __init__(self, solver: Solver, exact_solver=None, kernel='rbf', C=1.0, gamma='scale', degree=3, verbose=False):
        """

        :param solver: Inner solver for the optimization problem.
        :param exact_solver: Exact solver, should be used to verify or compare the results coming from the specified solver.
        :param kernel: Kernel type. the value should be taken from SVM.KERNELS.values
        :param C: Regularization parameter for the SVM problem
        :param gamma: Specify the gamma value for the rbf kernel. The parameter is ignored if kernel != "rbf"
        :param degree: Specify the degree for the polynomial kernel. The parameter is ignored if kernel != "poly"
        :param verbose: Enable prompts from the algorithm.
        """
        self.kernel_name = kernel
        self.verbose = verbose
        self.kernel = self._select_kernel(kernel)
        self.C = C
        self.gamma = gamma
        self.gamma_value = None
        self.solver = solver
        self.exact_solver = exact_solver

        self.degree = degree

    def create_rbf_kernel(self):
        if self.verbose:
            print("Rbf kernel with sigma = {}".format(self.gamma))
        return lambda x, y: np.exp(- self.gamma_value * np.square((np.linalg.norm(x - y))))

    def create_poly_kernel(self):
        if self.verbose:
            print("Polinomial kernel with grade {}".format(self.degree))
        return lambda x, xi: (self.gamma_value * np.inner(x, xi) + 1) ** self.degree

    def _select_kernel(self, kernel):
        if kernel == SVM.KERNELS['rbf']:
            return self.create_rbf_kernel()
        elif kernel == SVM.KERNELS['poly']:
            return self.create_poly_kernel()
        elif kernel == SVM.KERNELS['linear']:
            return self.create_poly_kernel(1)
        else:
            print("Not valid kernel name. Valid names are {}".format(SVM.KERNELS.values()))

    def compute_kernel_matrix(self, x):
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

        K = self.compute_kernel_matrix(x)
        Q = np.empty(K.shape)

        for i in range(n):
            for j in range(n):
                Q[i, j] = d[i] * d[j] * K[i, j]

        alpha = self.solve_optimization(d, Q)
        b = 0
        indexes = np.where(alpha > 1e-6)[0]
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

        :param d: Desired outputs vector
        :param Q: Matrix of the quadratic problem
        :return: Computed multipliers
        """
        n = Q.shape[0]
        q = - np.ones(n)
        l = np.full(n, 0.)
        u = np.full(n, float(self.C))

        self.solver.define_quad_objective(Q, q, l, u, d, 0)

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
        if self.verbose:
            print("PREDICTING...\nalpha={}".format(self.alpha))
        out = self.parallel_predict(x)
        if self.verbose:
            print(out)
        return np.sign(out)
