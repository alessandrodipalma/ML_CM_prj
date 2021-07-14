import time
import numpy
from joblib.numpy_pickle_utils import xrange

from gvpm import GVPM
from rosen import RosenGradientProjection
from svm import SVM, np
import cvxpy
import cplex


class SVR(SVM):

    def __init__(self, kernel='rbf', C=1, eps=0.001, gamma='scale', degree=3, solver='GVPM', tol=1e-3, plot_gap=False):
        super().__init__(kernel, C, gamma, degree)
        self.solver = solver
        self.eps = eps
        self.tol = tol
        self.plot_gap = plot_gap

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
        if self.plot_gap is True:
            alpha_opt, f_star, gradient = self.solve_with_cvxpy(2 * n, G, q, self.C, y, tol=self.tol)

        start_time = time.time()

        if self.solver == 'GVPM':

            alpha, gradient, proj_time, search_time = GVPM(G, q, l, u, y, e, n_min=1, plots=self.plot_gap, ls=GVPM.LS_EXACT) \
                .solve(x0=np.zeros(2 * n), max_iter=100, tol=self.tol, x_opt=alpha_opt, f_star=f_star)
            elapsed_time = time.time() - start_time

            if elapsed_time > 0:
                print("Elapsed time in GVPM {}\t{} % in projecting\t{} % in ls".format(elapsed_time,
                                                                                       proj_time * 100 / elapsed_time,
                                                                                       search_time * 100 / elapsed_time))
            print(alpha_opt)
            print(alpha)
            # bias = -np.mean(gradient * y)
            bias = 0
        elif self.solver == 'CPLEX':
            alpha, f_opt, gradient = self.solve_with_cvxpy(2 * n, G, q, self.C, y, tol=self.tol)
            # bias = -np.mean(gradient * y)
            bias = 0
        elif self.solver == 'rosen':
            alpha, gradient = RosenGradientProjection(G, q, l, u, y, e, lam_upp=1, lam_low=0.1).solve(
                x0=np.zeros(2 * n), max_iter=1000,
                min_d=1e-4, x_opt=alpha_opt, f_star=f_star)
            bias = 0
        end_time = time.time() - start_time
        print("took {} to solve with {}".format(end_time, self.solver))
        # ind = np.where(np.logical_and(0 <= alpha, alpha <= C))
        # print("ALPHAS", alpha)
        # print("sum: {}".format(np.sum(alpha * y)))
        # print(ind)

        print("bias={}".format(bias))
        # input()
        return alpha[:n] - alpha[n:], bias

    def compute_out(self, x):
        f = lambda i: self.support_alpha[i] * self.kernel(x, self.x[i]) + self.bias
        out = np.sum(np.array(list(map(f, np.arange(len(self.support_alpha))))))
        return out

    def predict(self, x):
        return np.array(list(map(self.compute_out, x)))

    def solve_with_cvxpy(self, n, G, q, C, y, tol):
        x = cvxpy.Variable(n)
        # x.value = x0
        objective = cvxpy.Minimize((1 / 2) * cvxpy.quad_form(x, G) + q.T @ x)
        constraints = [x >= 0, x <= C, y.T @ x == 0]
        problem = cvxpy.Problem(objective, constraints)
        problem.solve(verbose=True, solver="CPLEX", cplex_params={
            "barrier.convergetol": tol
        })
        # problem.backward()
        return numpy.array(x.value), problem.value, x.gradient
