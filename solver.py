from numpy.linalg import matrix_power


class Solver:
    def __init__(self, max_iter=100, tol=1e-3, verbose=False):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose


    def define_quad_objective(self, Q, q, left_constr, right_constr, y, b, ):
        self.Q = Q
        self.Q_square = matrix_power(Q, 2)
        self.q = q

        self.f = lambda x: 0.5 * (x.T @ Q @ x) + q @ x
        self.df = lambda x: Q @ x + q

        self.left_constr = left_constr
        self.right_constr = right_constr

        self.n = self.Q.shape[1]
        self.y = y
        self.b = b

    def solve(self, x0, x_opt=None, f_opt=None) -> tuple:
        pass