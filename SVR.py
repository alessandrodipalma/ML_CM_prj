import time
from Solver import Solver
from SVM import SVM, np
from joblib import Parallel, delayed
from SwapUtils import split_kernel, split_alpha, update_alpha, get_working_part, split_kernel_working


class SVR(SVM):

    def __init__(self, solver: Solver, exact_solver=None, decomp_solver: Solver = None, kernel='rbf', C=1, eps=1e-3,
                 gamma='scale', degree=3, alpha_tol = 1e-6, verbose=False):
        """
        :param solver: Inner solver for the optimization problem.
        :param exact_solver: Exact solver, should be used to verify or compare the results coming from the specified solver.
        :param decomp_solver: Solver for the SVR decomposition.
        :param kernel: Kernel type. the value should be taken from SVM.KERNELS.values
        :param C: Regularization parameter for the SVM problem
        :param eps: Epsilon parameter for the SVR problem.
        :param gamma: Specify the gamma value for the rbf kernel. The parameter is ignored if kernel != "rbf"
        :param degree: Specify the degree for the polynomial kernel. The parameter is ignored if kernel != "poly"
        """

        super().__init__(solver, exact_solver, kernel, C, gamma, degree, alpha_tol, verbose)
        self.eps = eps
        self.is_multi_output = False
        self.decompose = False
        self.decomp_solver = decomp_solver

    def fit(self, x, d):
        """

        :param x: Input vectors
        :param d: Desired outputs. Accepts also multidimensional outputs.
        :return: A tuple containing number of support vectors, the support multipliers, the indices of the support vectors in the original x vector. For multidimensional output problems, a tuples array is returned.
        """
        if len(x) == len(d):
            n = len(x)
        else:
            raise ValueError("X and y must have same size! Got X:{}, y:{}".format(x.shape, d.shape))

        self.compute_gamma(n, x)
        K = self.compute_kernel_matrix(x)

        if len(d.shape) == 1:
            self.x, self.support_alpha, self.d, self.bias, indexes = self.compute_alphas(K, d, n, x)
            return len(self.support_alpha), self.support_alpha, indexes
        else:  # train for each dimension
            self.dimensions = []
            self.is_multi_output = True
            self.dimensions = Parallel(n_jobs=len(d.shape), max_nbytes=None)(
                delayed(self.parallel_compute_alpha)(K, d, i, n, x) for i in range(d.shape[1]))
            self.dimensions = sorted(self.dimensions, key=lambda k: k['i'])
            return self.dimensions

    def compute_gamma(self, n, x):
        if self.gamma == 'auto':
            self.gamma_value = 1 / n
        elif self.gamma == 'scale':
            self.gamma_value = 1 / (n * x.var())

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
        indexes = np.where(abs(multipliers) > self.alpha_tol)[0]

        if self.verbose:
            print("number of sv: {}".format(len(indexes)))

        bias = self.compute_bias(alpha, d, n, x, multipliers)

        if self.verbose:
            print("bias {}".format(bias))
        x = x[indexes]
        support_alpha = multipliers[indexes]
        d = d[indexes]
        return x, support_alpha, d, bias, indexes

    def compute_bias(self, alpha, d, n, x, multipliers):
        """
        Compute the bias according to the formulation found in the haykin book.
        :param alpha:
        :param d:
        :param n:
        :param x:
        :param multipliers:
        :return:
        """
        def single_predict(input):
            return self.single_output_predition(input, multipliers, x, 0)

        predictions = np.array(list(map(single_predict, x)))
        estimates = d - predictions - np.full(len(x), - self.eps)
        a = alpha[:n]
        a_star = alpha[n:]
        precision = 1e-8 # use this instead of zero
        selected_estimates_left = estimates[np.where(np.logical_or(
            np.logical_and(a < self.C, a > precision),
            np.logical_and(a_star > precision, a_star < self.C)))]
        bias = np.mean(selected_estimates_left)
        return bias

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

        if self.decompose:
            # TODO make it work
            nsp = 128
            alpha = np.zeros(2 * n)

            def subproblem(k, alpha, working_indexes=None):
                if working_indexes is None:
                    qbb, qnb, qnn = split_kernel(Q, k * nsp, (k + 1) * nsp)

                    xb, xn = split_alpha(alpha, k * nsp, (k + 1) * nsp)
                    yb, yn = split_alpha(y, k * nsp, (k + 1) * nsp)
                    qb, qn = split_alpha(q, k * nsp, (k + 1) * nsp)
                    lb, ln = split_alpha(l, k * nsp, (k + 1) * nsp)
                    ub, un = split_alpha(u, k * nsp, (k + 1) * nsp)

                    Gbb = np.block([[qbb, -qbb], [-qbb, qbb]])
                    Gbn = np.block([[qnb, -qnb], [-qnb, qnb]])
                    eb = -yn @ xn
                    self.decomp_solver.define_quad_objective(Gbb, Gbn.T @ xn + qb, lb, ub, yb, eb)
                    # print("solving for k=", k)
                    xb, f_star, gradient = self.decomp_solver.solve(x0=xb)
                    # print("solved")
                    return xb, k
                else:
                    # print(Q.shape, working_indexes.shape)

                    effective = np.unique(
                        np.concatenate([working_indexes, working_indexes[:n] - n, working_indexes[n:] + n]))
                    ind_for_kern = np.unique(np.concatenate([working_indexes[n:], working_indexes[:n] - n]))
                    # print(ind_for_kern.shape, effective.shape)
                    qbb, qnb, qnn = split_kernel_working(Q, ind_for_kern)

                    xb, xn = get_working_part(alpha, effective)
                    yb, yn = get_working_part(y, effective)
                    qb, qn = get_working_part(q, effective)
                    lb, ln = get_working_part(l, effective)
                    ub, un = get_working_part(u, effective)

                    Gbb = np.block([[qbb, -qbb], [-qbb, qbb]])
                    Gbn = np.block([[qnb, -qnb], [-qnb, qnb]])
                    eb = -yn @ xn
                    print(Gbn.shape, xb.shape, xn.shape, alpha.shape, qb.shape)
                    self.decomp_solver.define_quad_objective(Gbb, Gbn.T @ xn + qb, lb, ub, yb, eb)
                    xb, f_star, gradient = self.decomp_solver.solve(x0=xb)
                    return xb, working_indexes

            working_indexes = None
            iter = 0
            while self.solver.grad_norm(alpha) > self.solver.tol and iter < 2:
                # print("gradient: ", )
                if working_indexes is None:
                    xbs = Parallel(n_jobs=4, max_nbytes=None)(
                        delayed(subproblem)(k, alpha) for k in range(int(n / nsp)))
                    for xb, k in xbs:
                        print(alpha.shape)
                        alpha = update_alpha(alpha, xb, k * nsp, (k + 1) * nsp)
                        print(alpha.shape)
                else:
                    xb, working_indexes = subproblem(0, alpha, working_indexes)
                    alpha[working_indexes] = xb

                working_indexes = np.where(alpha > self.alpha_tol)[0]
                iter += 1

            return alpha
        else:

            if self.verbose:
                start_time = time.time()

            alpha, f_star, gradient = self.solver.solve(x0=np.zeros(2 * n), x_opt=alpha_opt, f_opt=f_star)

            if self.verbose:
                end_time = time.time() - start_time
                print("took {} to solve".format(end_time))

            return alpha

    def compute_out(self, x):
        if self.is_multi_output:
            out = self.multi_output_prediction(x)
        else:
            out = self.single_output_predition(x, self.support_alpha, self.x, self.bias)

        return out

    def multi_output_prediction(self, x):
        out = np.zeros(len(self.dimensions))
        for i, dim in enumerate(self.dimensions):
            out[i] = self.single_output_predition(x, dim['support_alpha'], dim['x'], dim['bias'])
        return out

    def single_output_predition(self, x, support_alpha, sv, bias):
        f = lambda i: support_alpha[i] * self.kernel(x, sv[i])
        out = np.sum(np.array(list(map(f, np.arange(len(support_alpha)))))) + bias
        return out

    def predict(self, x):
        return self.parallel_predict(x)
