import cvxpy
import numpy

from solver import Solver


class CplexSolver(Solver):

    def solve(self, x0, **kwargs):
        x = cvxpy.Variable(self.n)
        # x.value = x0
        objective = cvxpy.Minimize((1 / 2) * cvxpy.quad_form(x, self.Q) + self.q.T @ x)
        constraints = [x >= self.left_constr[0], x <= self.right_constr[0], self.y.T @ x == 0]
        problem = cvxpy.Problem(objective, constraints)
        problem.solve(verbose=True, solver="CPLEX", cplex_params={
            "barrier.convergetol": self.tol
        })
        # problem.backward()
        return numpy.array(x.value), problem.value, x.gradient

