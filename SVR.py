from cvxopt.base import matrix

from gvpm import GVPM
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
        eps = np.full(n, self.eps)

        G = np.block([[Q, -Q],[-Q, Q]])
        q = np.concatenate((eps - d, eps + d))

        print(Q, "\n", G)
        A = np.concatenate((ide, np.diag(np.full(2*n, -1))))
        b = np.append(np.full(2*n, C), np.zeros(2*n))

        E = np.append(np.ones(n), -np.ones(n))
        E = E.reshape((1,E.shape[0]))
        e = np.zeros((1, 1))

        def f(x):
            return 0.5 * x @ G @ x + q.T @ x

        def df(x):
            return G @ x + q


        # alpha = GradientProjection(f=f, df=df, A=A, b=b, Q=E, q=e) \
        #     .solve(x0=np.zeros(2*n), maxiter=100)

        # ide = np.identity(n)
        # A = np.concatenate((ide, np.diag(np.full(n, -1))))
        # b = np.append(np.full(n, C), np.full(n, -C))
        alpha = GVPM(G, q, A=A, b=b).solve(x0= np.zeros(2*n))

        return - alpha[n:] + alpha[:n]

    def compute_out(self, x):
        f = lambda i: self.alpha[i] * self.kernel(x, self.x[i])
        out = np.sum(np.array(list(map(f, np.arange(len(self.alpha))))))
        # print(out)
        return out

    def predict(self, x):
        return np.array(list(map(self.compute_out, x)))