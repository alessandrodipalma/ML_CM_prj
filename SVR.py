from cvxopt.base import matrix

from svm import SVM, np, GradientProjection

class SVR(SVM):

    def __init__(self, kernel='rbf', C=1, eps=0.1, sigma=1, degree=3):
        super().__init__(kernel, C, sigma, degree)

        self.eps = eps

    def solve_optimization(self, C, d, n, Q):
        ide = np.identity(2*n)
        A = np.concatenate((ide, np.diag(np.full(2*n, -1))))
        b = np.append(np.full(2*n, C), np.zeros(2*n))
        E = np.append(np.ones(n), -np.ones(n))
        e = np.zeros((1, 1))

        def f(x):
            diff = x[:n] - x[n:]
            sum = x[:n] + x[n:]

            return 0.5 * diff.T @ Q @ diff - d.T @ diff + self.eps * np.sum(sum)

        def df(x):
            diff = x[:n] - x[n:]
            eps = np.full(n, self.eps)
            Q_dot_diff = Q @ diff
            da = - d + eps + Q_dot_diff
            da_p = d + eps - Q_dot_diff

            return np.append(da, da_p)

        alpha = GradientProjection(f=f, df=df, A=A, b=b, Q=E.reshape((1, E.shape[0])), q=e) \
            .solve(delta_d=0.001, maxiter=100)

        return alpha[:n] - alpha[n:]

    def predict(self, x):
        out = np.array(list(map(self.compute_out, x)))
        return out




    