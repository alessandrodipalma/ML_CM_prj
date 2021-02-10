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

        alpha = self.solve_optimization(C, d, n, K)
        # print(alpha)
        # self.gradient = Q @ alpha + d

        indexes = np.where(np.abs(alpha) > 1e-6)[0]
        # print("multipliers: ", alpha[indexes])
        self.alpha = alpha[indexes]
        self.x = x[indexes]
        self.d = d[indexes]

        return len(self.alpha), self.alpha, []

    def solve_optimization(self, C, d, n, Q):
        ide = np.identity(2*n)
        eps = np.full(n,- self.eps)

        G = np.block([[Q, -Q],[-Q, Q]])
        q = np.concatenate((eps - d, eps + d))

        # print(Q, "\n", G)
        A = np.concatenate((ide, -ide))
        b = np.append(np.full(2*n, C), np.zeros(2*n))

        y = np.append(np.ones(n), -np.ones(n))
        E = y.reshape((1,y.shape[0]))
        e = np.zeros((1, 1))
        
        # alpha = GradientProjection(f=lambda x: 0.5 * x.T @ G @ x + q.T @ x, df=lambda x: G @ x + q,A=A, b=b).solve(x0=np.zeros(2*n), maxiter=25)

        # alpha = qp(matrix(G), matrix(q), G=matrix(A), h=matrix(b))
        # alpha = np.array(alpha['x']).ravel()

        alpha = GVPM(G, q, np.zeros(2*n), np.full(2*n, C)).solve(x0=np.zeros(2*n), max_iter=100, min_d=1e-2)

        return alpha[:n] - alpha[n:]

    def compute_out(self, x):
        f = lambda i: self.alpha[i] * self.kernel(x, self.x[i])
        out = np.sum(np.array(list(map(f, np.arange(len(self.alpha))))))
        return out

    def predict(self, x):
        return np.array(list(map(self.compute_out, x)))

