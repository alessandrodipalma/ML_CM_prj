from cvxopt.base import matrix
from scipy.optimize import optimize, minimize

from gvpm import GVPM
from cvxopt import matrix
from cvxopt.solvers import qp
from svm import SVM, np, GradientProjection


class SVR(SVM):

    def __init__(self, kernel='rbf', C=1, eps=0.001, sigma=1, degree=3):
        super().__init__(kernel, C, sigma, degree)

        self.eps = eps

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

        alpha, self.bias = self.solve_optimization(C, d, n, K)
        # print(alpha)
        # self.gradient = Q @ alpha + d

        indexes = np.where(alpha > C/10000)[0]
        # print("multipliers: ", alpha[indexes])
        self.alpha = alpha[indexes]
        self.x = x[indexes]
        self.d = d[indexes]

        return len(self.alpha), self.alpha, []

    def solve_optimization(self, C, d, n, Q):
        ide = np.identity(2 * n)
        eps = np.full(n, - self.eps)

        G = np.block([[Q, -Q], [-Q, Q]])
        q = np.concatenate((eps - d, eps + d))

        # box constraints
        l = np.full(2 * n, 0.)
        u = np.full(2 * n, float(C))

        # knapsack constraint
        y = np.append(np.full(n, 1.), np.full(n, -1.))
        e = np.full((1, 1), 0.)

        # x0 = np.random.uniform(low=0.01, high=C * 0.99, size=(2 * n,))
        x0 = np.zeros(2*n)
        # print(Q, "\n", G)

        # cvxopt solver
        A = np.concatenate((ide, -ide))
        b = np.append(np.full(2*n, C), np.zeros(2*n))
        # E = y.reshape((1, y.shape[0]))

        out = qp(matrix(G), matrix(q), G=matrix(A), h=matrix(b))
        alpha = np.array(out['x']).ravel()

        # alpha, gradient = GVPM(G, q, l, u, y, e).solve(x0=x0, max_iter=100, min_d=1e-6)

        # gradient = G @ alpha + q
        # ind = np.where(np.logical_and(0 <= alpha, alpha <= C))
        print(alpha)
        print("sum: {}".format(np.sum(alpha * y)))
        # print(ind)
        bias = -np.mean(gradient*y)
        # bias = 0
        # bias = 0

        print("bias={}".format(bias))
        # input()
        return alpha[:n] - alpha[n:], bias

    def compute_out(self, x):
        f = lambda i: self.alpha[i] * self.kernel(x, self.x[i]) + self.bias
        out = np.sum(np.array(list(map(f, np.arange(len(self.alpha))))))
        return out

    def predict(self, x):
        return np.array(list(map(self.compute_out, x)))
