

"""
Line-search strategy comparison: EXACT vs BACKTRACKING
Plots: GAP
Data: N_iter, ls_cost, time_tot, time_ls
"""
import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from Cplex_Solver import CplexSolver
from SVR import SVR
from GVPM import GVPM


feature_samples_dict = [{'features': 10, 'samples': 200},
                        {'features': 100, 'samples': 200},
                        {'features': 100, 'samples': 500},
                        {'features': 100, 'samples': 1000},
                        {'features': 300, 'samples': 1000},
                        {'features': 1000, 'samples': 2000}
                        ]

histories = []
table = []
plt.rcParams["figure.figsize"] = (10, 20)
fig, axs = plt.subplots(3,2)
plt.yscale('log')
problems = 1

for i, d in enumerate(feature_samples_dict):
    C = 1
    kernel = 'rbf'
    eps = 0.1
    gamma = 'scale'
    tol = 1e-3

    row = {'features': d['features'], 'samples': d['samples']}
    histories = {}

    plot = axs[int(i / 2), i % 2]

    for ls in GVPM.LineSearches.values:
        solver = GVPM(ls=ls, n_min=2, tol=tol, lam_low=1e-3, plots=False, proj_tol=1e-3)
        stats = []

        for p in range(problems):
            X, y = make_regression(n_samples=d['samples'], n_features=d['features'])
            X = preprocessing.StandardScaler().fit(X).transform(X)
            y = 2 * (y - min(y)) / (max(y) - min(y)) - 1
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

            model = SVR(solver=solver, C=C, kernel=kernel, eps=eps, gamma=gamma,
                        exact_solver=CplexSolver(tol=1e-10, verbose=False))
            n_sv, alphas, indices = model.train(X_train, y_train)
            stats.append(solver.stats)

        final_stats = {}
        for k in stats[0].keys():
            v = 0
            for s in stats:
                v += s[k]
            # final_stats =
            row[ls + ' ' + k] = v/problems

        plot.plot(np.arange(len(solver.f_gap_history)), solver.f_gap_history, label=ls)
    plot.set_title("{} features, {} samples".format(d["features"], d["samples"]))
    plot.legend()
    plot.set_yscale('log')

    table.append(row)



out_dir = "plots/line_search_bounds/"
os.mkdir(out_dir)
plt.savefig(out_dir + "ls.png")
with open(out_dir + "ls_table.txt", "w", encoding="utf-8") as out_file:
    out_file.write(tabulate([r.values() for r in table], table[0].keys(), tablefmt='latex', floatfmt=".2e"))
with open(out_dir + "ls_table.csv", "w", encoding="utf-8") as out_file:
    out_file.write(tabulate([r.values() for r in table], table[0].keys(), tablefmt='csv', floatfmt=".2e"))








