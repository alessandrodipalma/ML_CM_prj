"""
How the lambda minimum value affects convergence (probably noticeable only on big problems)

"""
from utils import generate_regression_from_feature_sample_dict

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
                        {'features': 50, 'samples': 200},
                        {'features': 100, 'samples': 100},
                        {'features': 100, 'samples': 500},
                        {'features': 200, 'samples': 500},
                        {'features': 300, 'samples': 1000},
                        ]

lambdas = [1e-1, 1e-3, 1e-8]

histories = []
table = []
plt.rcParams["figure.figsize"] = (20, 20)
plt.title("Lambda Variation")
cols = 3
fig, axs = plt.subplots(2,cols)
plt.yscale('log')

n_problems = 50
all_problems = generate_regression_from_feature_sample_dict(feature_samples_dict, n_problems)

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

    for ls in GVPM.LineSearches.values:
        for t, lam in enumerate(lambdas):
            solver = GVPM(ls=ls, n_min=2, tol=tol, lam_low=lam, plots=False, proj_tol=1e-2)
            stats = []

            for p in all_problems[i]:
                X_train, X_test, y_train, y_test = p

                model = SVR(solver=solver, C=C, kernel=kernel, eps=eps, gamma=gamma,
                            exact_solver=CplexSolver(tol=tol, verbose=False))
                n_sv, alphas, indices = model.train(X_train, y_train)
                stats.append(solver.stats)

            it = 0
            f_gap = 0
            time_spent = 0
            for s in stats:
                it += s['it']
                f_gap += solver.f_gap_history[-1]
                time_spent += s['time_tot']
            # final_stats =
            row["{} {} it".format(ls, lam)] = it / n_problems
            row["{} {} f_gap".format(ls, lam)] = f_gap / n_problems
            row["{} {} time".format(ls, lam)] = time_spent / n_problems

            plot.plot(np.arange(len(solver.f_gap_history)), solver.f_gap_history, label='min stepsize: {}, {} ls'.format(lam,ls))
    plot.set_title(d)
    plot.legend()
    plot.set_yscale('log')

    table.append(row)

out_dir = "plots/lambda/"
plt.savefig(out_dir + "lambda_f_gap.png")
with open(out_dir + "table.txt", "w", encoding="utf-8") as out_file:
    out_file.write(tabulate([r.values() for r in table], table[0].keys(), tablefmt='latex'))

