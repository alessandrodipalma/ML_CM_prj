"""
How the lambda minimum value affects convergence (probably noticeable only on big problems)

"""
import os

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
from GVPM import GVPM


feature_samples_dict = [{'features': 10, 'samples': 200},
                        {'features': 50, 'samples': 200},
                        {'features': 100, 'samples': 100},
                        {'features': 200, 'samples': 300},
                        {'features': 200, 'samples': 500},
                        {'features': 300, 'samples': 1000},
                        ]

# lambdas = [1e-1, 1e-3, 1e-8]

histories = []
table = []
plt.rcParams["figure.figsize"] = (20, 15)
cols = 3
fig, axs = plt.subplots(2,cols)
plt.yscale('log')

n_problems = 10
all_problems = generate_regression_from_feature_sample_dict(feature_samples_dict, n_problems)

for ls in GVPM.LineSearches.values:
    plt.rcParams["figure.figsize"] = (20, 15)
    cols = 3
    fig, axs = plt.subplots(2, cols)
    plt.yscale('log')

    for i, d in enumerate(feature_samples_dict):
        C = 1
        kernel = 'rbf'
        eps = 0.1
        gamma = 'scale'
        tol = 1e-2

        row = {'features': d['features'], 'samples': d['samples']}
        histories = {}

        plot = axs[int(i / cols), i % cols]


        for lam in [1, 1e-1, 1e-2, 1e-3]:
            solver = GVPM(ls=ls, n_min=2, tol=tol, lam_low=lam, plots=False, proj_tol=1e-2, max_iter=100, do_stats=True)
            stats = []

            for p in all_problems[i]:
                X, y = p

                model = SVR(solver=solver, C=C, kernel=kernel, eps=eps, gamma=gamma,
                            exact_solver=CplexSolver(tol=1e-10, verbose=False))
                n_sv, alphas, indices = model.train(X, y)
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

        plot.set_title("{} features, {} samples".format(d["features"], d["samples"]))
        plot.legend()
        plot.set_yscale('log')

        table.append(row)

    out_dir = "plots/lambda_retry/"
    try:
        os.mkdir(out_dir)
    except Exception:
        pass

    plt.savefig(out_dir + "lambda_{}.png".format(ls))
    with open(out_dir + "lamda_{}.txt".format(ls), "w", encoding="utf-8") as out_file:
        out_file.write(tabulate([r.values() for r in table], table[0].keys(), tablefmt='latex', floatfmt=".2e"))
    with open(out_dir + "lamda_{}.csv".format(ls), "w", encoding="utf-8") as out_file:
        out_file.write(tabulate([r.values() for r in table], table[0].keys(), tablefmt='tsv', floatfmt=".2e"))

