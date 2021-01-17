import numpy as np
from gradientprojection import GradientProjection

n = 10

A = np.random.rand(n, n)
Q = A @ A.transpose()

# test case from Nonlinear programming book 10.5.5

f = lambda x: 2*(x[0]**2) + 2 * (x[1]**2) - 2*x[0]*x[1] - 4*x[0] - 6*x[1]
def df(x):
    dx1 = lambda x: 4*x[0] - 2*x[1] - 4
    dx2 = lambda x: 4*x[1] - 2*x[0] - 6

    return np.array([dx1(x), dx2(x)])

A = np.array([[1,1],[1,5],[-1,0],[0,-1]])
b = np.array([2,5,0,0])


opt = GradientProjection(f=f, df=df, A=A, b=b, x0=np.array([0,0])).solve()
print("Optimum value is: {}".format(opt))
