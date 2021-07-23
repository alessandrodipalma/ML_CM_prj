from numpy.linalg import matrix_power


class Solver:


    def __init__(self, max_iter=100, tol=1e-3, verbose=False):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self._objective_defined = False


    def define_quad_objective(self, Q, q, left_constr, right_constr, y, b):
        """
        :param Q: nxn matrix
        :param q: an n-sized vector
        :param left_constr: n-sized vector defining the left constraint on each element of x
        :param right_constr: n-sized vector defining the right constraint on each element of x
        :param y: vector associated with the linear constraint s.t. y.T @ x = b
        :param b: vector associated with the linear constraint s.t. y.T @ x = b
        :return:
        """
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

        self._objective_defined = True

    def solve(self, x0, x_opt=None, f_opt=None) -> tuple:
        if not self._objective_defined:
            raise Exception("Must define objective function first")

    def get_solving_stats(self):
        return self.stats
