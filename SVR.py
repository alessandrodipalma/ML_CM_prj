from cvxopt.base import matrix

from svm import SVM, np, GradientProjection

class SVR(SVM):

    def __init__(self, kernel='rbf', C=1, eps=0.01, sigma=1, degree=3):
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
        # print("Kernel:", K)
        Q = np.empty(K.shape)

        for i in range(n):
            for j in range(n):
                Q[i, j] = d[i] * d[j] * K[i, j]

        alpha = self.solve_optimization(C, d, n, Q)
        # print(alpha)

        indexes = np.where(alpha != 0)[0]
        print("multipliers: ", alpha[indexes])
        self.alpha = np.array(alpha[indexes])

        self.x = np.array(x[indexes])

        return len(self.alpha), self.alpha, indexes

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


        alpha = GradientProjection(f=f, df=df, A=A, b=b) \
            .solve(x0=np.zeros(2*n), delta_d=0.00001, maxiter=50, c=1e-6)

        return alpha[:n] - alpha[n:]

    def compute_out(self, x):
        f = lambda i: self.alpha[i] * self.kernel(x, self.x[i])
        out = np.sum(np.array(list(map(f, np.arange(len(self.alpha))))))
        # print(out)
        return out

    def predict(self, x):
        return np.array(list(map(self.compute_out, x)))