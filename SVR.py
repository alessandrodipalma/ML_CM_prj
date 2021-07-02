import numpy
from cvxopt.base import matrix
from scipy.optimize import optimize, minimize

from gvpm import GVPM
from cvxopt import matrix
from cvxopt.solvers import qp
from svm import SVM, np, GradientProjection
import cvxpy
import cplex

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

        self.alpha, self.bias = self.solve_optimization(C, d, n, K)
        print("COMPUTED MULTIPLIERS: {}".format(self.alpha))
        # self.gradient = Q @ alpha + d

        indexes = np.where(abs(self.alpha) > (C/100))[0]
        print("number of sv: ", len(indexes),)
        self.x = x[indexes]
        self.support_alpha = self.alpha[indexes]
        self.d = d[indexes]
        # input()
        return len(self.alpha), self.alpha, indexes

    def solve_optimization(self, C, d, n, Q):
        eps = np.full(n, - self.eps)

        G = np.block([[Q, -Q], [-Q, Q]])
        q = np.concatenate((eps - d, eps + d))

        # box constraints
        l = np.full(2 * n, 0.)
        u = np.full(2 * n, float(C))

        # knapsack constraint
        y = np.append(np.full(n, 1.), np.full(n, -1.))
        e = np.full((1, 1), 0.)

        # alpha0 = np.random.uniform(low=0.01, high=C * 0.99, size=(2 * n,))
        try:
            alpha0 = self.alpha
        except AttributeError:
            self.alpha = np.zeros(2 * n)
            alpha0 = self.alpha

        # alpha = self.solve_with_cvxpy(2*n, G, q, C, y, alpha0)
        # gradient = G @ alpha + q


        alpha, gradient = GVPM(G, q, l, u, y, e).solve(x0=np.zeros(2*n), max_iter=100, min_d=1e-6)

        # ind = np.where(np.logical_and(0 <= alpha, alpha <= C))
        print("ALPHAS", alpha)
        print("sum: {}".format(np.sum(alpha * y)))
        # print(ind)
        bias = np.mean(gradient*y)

        print("bias={}".format(bias))
        # input()
        return alpha[:n] - alpha[n:], bias

    def compute_out(self, x):
        f = lambda i: self.support_alpha[i] * self.kernel(x, self.x[i]) + self.bias
        out = np.sum(np.array(list(map(f, np.arange(len(self.support_alpha))))))
        return out

    def predict(self, x):
        return np.array(list(map(self.compute_out, x)))

    def solve_with_cvxpy(self, n, G, q, C, y, x0):
        x = cvxpy.Variable(n)
        # x.value = x0
        objective = cvxpy.Minimize((1/2)*cvxpy.quad_form(x, G) + q.T @ x)
        constraints = [x >= 0, x <= C, y.T @ x == 0]
        problem = cvxpy.Problem(objective, constraints)
        problem.solve(solver=cvxpy.CPLEX)
        return numpy.array(x.value)
