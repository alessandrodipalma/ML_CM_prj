"""
How the range [a_min, a_max] affects convergence
ranges: +- 1e-2 , -+ 1e-6 , +- 1e-12
"""
from sklearn import preprocessing
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from utils import generate_regression_from_feature_sample_dict
import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate

from Cplex_Solver import CplexSolver
from SVR import SVR
from GVPM import GVPM


np.random.seed(42)
n_features = 200
n_samples = 200
X, y = make_regression(n_samples=n_samples, n_features=n_features)

X = preprocessing.StandardScaler().fit(X).transform(X)
y = 2 * (y - min(y)) / (max(y) - min(y)) - 1

print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


out_dir = "plots/epochs/"
histories = []
table = []
plt.rcParams["figure.figsize"] = (10, 10)


plt.title("Gap through epochs")
plt.yscale('log')

batch_size = int(len(X_train) / 5)
print(batch_size)



train_err = []
test_err = []


def mse(prediction, y_train):
    pass


for i in range(int(n_samples/batch_size)):
    bs = (i+1)*batch_size
    plt.title("{} samples".format(bs))
    plt.yscale('log')
    for n_min in [0,2,4,8,16]:
        C = 1
        kernel = 'rbf'
        eps = 0.1
        gamma = 'scale'
        tol = 1e-3
        ls = GVPM.LineSearches.BACKTRACK
        solver = GVPM(ls=ls, n_min=n_min, tol=tol, lam_low=1e-3, plots=False, proj_tol=1e-3)
        model = SVR(solver=solver, C=C, kernel=kernel, eps=eps, gamma=gamma,
                    exact_solver=CplexSolver(tol=tol, verbose=False))

        histories = {}

        stats = []
        n_sv, alphas, indices = model.train(X_train[:bs], y_train[:bs])
        prediction = model.predict(X_train)

        train_err.append(mse(prediction, y_train))
        test_err.append(mse(model.predict(X_test), y_test))

        plt.plot(np.arange(len(solver.f_gap_history)), solver.f_gap_history, label='n_min {}'.format(n_min))
    plt.legend()
    plt.show()
# plt.savefig(out_dir + "gap_bt_ls.png")
# with open(out_dir + "table_bt_ls.txt", "w", encoding="utf-8") as out_file:
#     out_file.write(tabulate([r.values() for r in table], table[0].keys(), tablefmt='latex'))

