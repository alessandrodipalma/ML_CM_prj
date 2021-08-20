import time
from solver import Solver
from svm import SVM, np
from joblib import Parallel, delayed

class SVR(SVM):

    def __init__(self, solver: Solver, exact_solver=None, kernel='rbf', C=1, eps=0.001, gamma='scale', degree=3):
        super().__init__(solver, exact_solver, kernel, C, gamma, degree)

        self.eps = eps
        self.is_multi_output = False

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
        print(d.shape, len(d.shape))
        if len(d.shape) == 1:
            self.x, self.support_alpha, self.d, self.bias, indexes = self.compute_alphas(K, d, n, x)
            return len(self.support_alpha), self.support_alpha, indexes
        else: # train for each dimension
            self.dimensions = []
            self.is_multi_output = True
            self.dimensions = Parallel(n_jobs=2)(delayed(self.parallel_compute_alpha)(K, d, i, n, x) for i in range(d.shape[1]))
            self.dimensions = sorted(self.dimensions, key=lambda k: k['i'])
            return self.dimensions

    def parallel_compute_alpha(self, K, d, i, n, x):
        # print("----------------------------------------------- i = ", i)
        support_x, support_alpha, des_out, bias, support_indexes = self.compute_alphas(K, d[:, i], n, x)
        return {
            'i': i,
            'x': support_x,
             'support_alpha': support_alpha,
             'd': des_out,
             'bias': bias,
             'indexes': support_indexes
             }

    def compute_alphas(self, K, d, n, x):
        alpha = self.solve_optimization(d, K)
        multipliers = alpha[:n] - alpha[n:]
        indexes = np.where(abs(multipliers) > (self.C * 1e-6))[0]
        # print("number of sv: ", len(indexes), )

        def single_predict(input):
            return self.single_output_predition(input, multipliers, x, 0)

        predictions = np.array(list(map(single_predict, x)))
        estimates = d - predictions - np.full(len(x), - self.eps)

        a = alpha[:n]
        a_star = alpha[n:]
        selected_estimates_left = estimates[np.where(np.logical_or(np.logical_and(a < self.C, a > 1e-6),
                                                                   np.logical_and(a_star > 1e-6, a_star < self.C)))]

        bias = np.mean(selected_estimates_left)
        if self.verbose:
            print("bias {}".format(bias))
        x = x[indexes]
        support_alpha = multipliers[indexes]
        d = d[indexes]
        return x, support_alpha, d, bias, indexes

    # def comput_bias(self):
    #     i0 = np.where(np.)

    def solve_optimization(self, d, Q):
        """
        :param d: desired outputs
        :param Q: Computed kernel matrix
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
        alpha, f_star, gradient = self.solver.solve(x0=np.zeros(2 * n), x_opt=alpha_opt, f_opt=f_star)
        end_time = time.time() - start_time

        if self.verbose:
            print("took {} to solve".format(end_time))
        # input()
        return alpha

    def compute_out(self, x):
        if self.is_multi_output:
            out = np.zeros(len(self.dimensions))
            for i, dim in enumerate(self.dimensions):
                out[i] = self.single_output_predition(x, dim['support_alpha'], dim['x'], dim['bias'])
        else:
            out = self.single_output_predition(x, self.support_alpha, self.x, self.bias)

        return out

    def single_output_predition(self, x, support_alpha, sv, bias):
        f = lambda i: support_alpha[i] * self.kernel(x, sv[i])
        out = np.sum(np.array(list(map(f, np.arange(len(support_alpha)))))) \
              + bias
        return out

    def predict(self, x):
        return self.parallel_predict(x)
