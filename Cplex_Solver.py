import time

import cvxpy
import numpy

from Solver import Solver


class CplexSolver(Solver):

    def solve(self, x0, x_opt=None, f_opt=None):
        super().solve(x0, x_opt, f_opt)
        x = cvxpy.Variable(self.n)
        objective = cvxpy.Minimize((1 / 2) * cvxpy.quad_form(x, self.Q) + self.q.T @ x)
        constraints = [x >= self.left_constr[0], x <= self.right_constr[0], self.y.T @ x == 0]
        problem = cvxpy.Problem(objective, constraints)

        start_time = time.time()
        problem.solve(verbose=self.verbose, solver="CPLEX", cplex_params={
            "barrier.convergetol": self.tol
        })
        self.f_value = problem.value
        self.x_value = numpy.array(x.value)
        self.elapsed_time = time.time() - start_time
        self.iterations = problem.solver_stats.num_iters
        return numpy.array(x.value), problem.value, x.gradient
