import time
import numpy
from joblib.numpy_pickle_utils import xrange

from gvpm import GVPM
from rosen import RosenGradientProjection
from solver import Solver
from svm import SVM, np
import cvxpy
import cplex


class SVR(SVM):

    def __init__(self, solver: Solver, exact_solver=None, kernel='rbf', C=1, eps=0.001, gamma='scale', degree=3):
        super().__init__(kernel, C, gamma, degree)
        self.solver = solver
        self.exact_solver = exact_solver
        self.eps = eps

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
        # print("training with x={}, d={}".format(x,d))

        K = self.compute_K(x)
        alpha, self.bias = self.solve_optimization(d, K)
        # print("COMPUTED MULTIPLIERS: {}".format(alpha))
        # self.gradient = Q @ alpha + d

        indexes = np.where(abs(alpha) > (self.C * 1e-6))[0]
        print("number of sv: ", len(indexes), )
        self.x = x[indexes]
        self.support_alpha = alpha[indexes]
        self.d = d[indexes]
        # input()
        return len(self.support_alpha), self.support_alpha, indexes

    def solve_optimization(self, d, Q):
        """
        :param d: desired outputs
        :param n:
        :param Q: Computed kernel matrix
        :param solver:
        :param knapsack_solver:
        :return:
        """
        n = Q.shape[0]
        eps = np.full(n, - self.eps)

        G = np.block([[Q, -Q], [-Q, Q]])
        q = np.concatenate((eps - d, eps + d))

        # box constraints
        l = np.full(2 * n, 0.)
        u = np.full(2 * n, float(self.C))

        # knapsack constraint
        y = np.append(np.full(n, 1.), np.full(n, -1.))
        e = np.full((1, 1), 0.)

        f_star = alpha_opt = None
        if self.exact_solver is not None:
            self.exact_solver.define_quad_objective(G, q, l, u, y, e)
            alpha_opt, f_star, gradient = self.exact_solver.solve(x0=np.zeros(2 * n), x_opt=alpha_opt, f_opt=f_star)

        self.solver.define_quad_objective(G, q, l, u, y, e)
        start_time = time.time()
        print(alpha_opt, f_star)
        alpha, f_star, gradient = self.solver.solve(x0=np.zeros(2 * n), x_opt=alpha_opt, f_opt=f_star)
        end_time = time.time() - start_time

        bias = 0

        print("took {} to solve".format(end_time))
        print("bias={}".format(bias))
        # input()
        return alpha[:n] - alpha[n:], bias

    def compute_out(self, x):
        f = lambda i: self.support_alpha[i] * self.kernel(x, self.x[i]) + self.bias
        out = np.sum(np.array(list(map(f, np.arange(len(self.support_alpha))))))
        return out

    def predict(self, x):
        return np.array(list(map(self.compute_out, x)))
