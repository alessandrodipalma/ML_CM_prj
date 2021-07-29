"""
How n_min affects convergence stability
n values: 0,1,2,4,8
"""
import os

from utils import generate_regression_from_feature_sample_dict
import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate

from Cplex_Solver import CplexSolver
from SVR import SVR
from gvpm import GVPM


feature_samples_dict = [
                        {'features': 10, 'samples': 200},
                        {'features': 100, 'samples': 50},
                        {'features': 50, 'samples': 200},
                        # {'features': 100, 'samples': 200},
                        # {'features': 100, 'samples': 500},
                        # {'features': 100, 'samples': 1000},
                        # {'features': 300, 'samples': 1000},
                        # {'features': 500, 'samples': 2000},
                        ]
out_dir = "./plots/n_min/"
# os.mkdir("./plots")
# os.mkdir("./plots/n_min/")
tols = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

histories = []
table = []
plt.rcParams["figure.figsize"] = (15, 30)
# plt.title("n_min Variation")
cols = 1
fig, axs = plt.subplots(3,cols)
plt.yscale('log')

n_problems = 1
all_problems = generate_regression_from_feature_sample_dict(feature_samples_dict, n_problems, fixed_rs=42)
tols.reverse()
for i, d in enumerate(feature_samples_dict):
    C = 1
    kernel = 'rbf'
    eps = 0.1
    gamma = 'scale'
    # tol = 1e-5
    ls = GVPM.LineSearches.BACKTRACK

    row = {'features': d['features'], 'samples': d['samples']}
    histories = {}

    plot = axs[i]

    for t, tol in enumerate(tols):
        solver = GVPM(ls=ls, n_min=2, tol=tol, lam_low=1e-3, plots=False, proj_tol=1e-8, max_iter=2000)
        stats = []

        for p in all_problems[i]:
            X, y = p

            model = SVR(solver=solver, C=C, kernel=kernel, eps=eps, gamma=gamma,
                        exact_solver=CplexSolver(tol=1e-12, verbose=False))
            n_sv, alphas, indices = model.train(X, y)
            stats.append(solver.stats)

        final_stats = {}
        it = 0
        f_gap = 0
        time_spent = 0
        for s in stats:
            it += s['it']
            f_gap += solver.f_gap_history[-1]
            time_spent += s['time_tot']
        # final_stats =
        # row["{} it".format(n_min)] = it / n_problems
        # row["{} f_gap".format(n_min)] = f_gap / n_problems
        # row["{} time".format(n_min)] = time_spent / n_problems

        plot.plot(np.arange(len(solver.f_gap_history)), solver.f_gap_history, label='tol = {}'.format(tol), linewidth=4)
    plot.set_title("{} features, {} samples".format(d['features'], d['samples']))
    plot.legend()
    plot.set_yscale('log')

    table.append(row)

plt.show()
# plt.savefig(out_dir + "n_min_f_gap_large_problems.png")
# with open(out_dir + "table_large_problems.txt", "w", encoding="utf-8") as out_file:
#     out_file.write(tabulate([r.values() for r in table], table[0].keys(), tablefmt='latex'))

