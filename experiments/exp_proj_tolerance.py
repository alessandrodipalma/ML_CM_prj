
"""
How the projection precision affects the global convergence
values: 1e-1, 1e-2, 1e-4, 1e-8
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from Cplex_Solver import CplexSolver
from SVR import SVR
from gvpm import GVPM


feature_samples_dict = [{'features': 10, 'samples': 200},
                        {'features': 100, 'samples': 200},
                        {'features': 100, 'samples': 500},
                        {'features': 100, 'samples': 1000},
                        {'features': 300, 'samples': 1000},
                        {'features': 1000, 'samples': 2000}
                        ]
proj_tols = [1e-1, 1e-2, 1e-4, 1e-8]

histories = []
table = []
plt.rcParams["figure.figsize"] = (20, 20)
cols = 3
fig, axs = plt.subplots(2,cols)
plt.yscale('log')

for i, d in enumerate(feature_samples_dict):
    C = 1
    kernel = 'rbf'
    eps = 0.1
    gamma = 'scale'
    tol = 1e-3
    ls = GVPM.LineSearches.BACKTRACK

    row = {'features': d['features'], 'samples': d['samples']}
    histories = {}

    plot = axs[int(i / cols), i % cols]

    for t, pr_tol in enumerate(proj_tols):
        solver = GVPM(ls=ls, n_min=2, tol=tol, lam_low=1e-3, plots=False, proj_tol=pr_tol)
        stats = []

        problems = 50

        for p in range(problems):
            X, y = make_regression(n_samples=d['samples'], n_features=d['features'])
            X = preprocessing.StandardScaler().fit(X).transform(X)
            y = 2 * (y - min(y)) / (max(y) - min(y)) - 1
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

            model = SVR(solver=solver, C=C, kernel=kernel, eps=eps, gamma=gamma,
                        exact_solver=CplexSolver(tol=tol, verbose=False))
            n_sv, alphas, indices = model.train(X_train, y_train)
            stats.append(solver.stats)

        final_stats = {}
        v = 0
        for s in stats:
            v += s['it']
        # final_stats =
        row[pr_tol] = v / problems

        plot.plot(np.arange(len(solver.f_gap_history)), solver.f_gap_history, label='projection tolerance: {}'.format(pr_tol))
    plot.set_title(d)
    plot.legend()
    plot.set_yscale('log')

    table.append(row)

plt.show()
print(tabulate([r.values() for r in table], table[0].keys(), tablefmt='fancy_grid'))

