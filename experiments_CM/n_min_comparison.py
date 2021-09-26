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
from GVPM import GVPM


feature_samples_dict = [
                        {'features': 10, 'samples': 200},
                        {'features': 100, 'samples': 50},
                        # {'features': 50, 'samples': 200},
                        {'features': 100, 'samples': 200},
                        {'features': 100, 'samples': 500},
                        {'features': 100, 'samples': 1000},
                        {'features': 300, 'samples': 1000},
                        # {'features': 500, 'samples': 2000},
                        ]
out_dir = "./plots/n_min/"
# os.mkdir("./plots")
# os.mkdir("./plots/n_min/")
n_mins = [0,1,3,8,15]

histories = []
table = []
plt.rcParams["figure.figsize"] = (15, 15)
# plt.title("n_min Variation")
cols = 2
rows = 3
fig, axs = plt.subplots(rows,cols)
plt.yscale('log')

n_problems = 20
all_problems = generate_regression_from_feature_sample_dict(feature_samples_dict, n_problems, fixed_rs=42)

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

    for t, n_min in enumerate(n_mins):
        solver = GVPM(ls=ls, n_min=n_min, tol=tol, lam_low=1e-3, plots=False, proj_tol=1e-3, stopping_rule=GVPM.StoppingRules.gradient)
        stats = []

        for p in all_problems[i]:
            X, y = p

            model = SVR(solver=solver, C=C, kernel=kernel, eps=eps, gamma=gamma,
                        exact_solver=CplexSolver(tol=1e-10, verbose=False))
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
        row["{} it".format(n_min)] = it / n_problems
        row["{} f_gap".format(n_min)] = f_gap / n_problems
        row["{} time".format(n_min)] = time_spent / n_problems

        plot.plot(np.arange(len(solver.f_gap_history)), solver.f_gap_history, label='n_min = {}'.format(n_min))
    plot.set_title("{} features, {} samples".format(d['features'], d['samples']))
    plot.legend()
    plot.set_yscale('log')

    table.append(row)

# plt.show()
plt.savefig(out_dir + "n_min_stop_d.png")
with open(out_dir + "n_min_table_stop_d.txt", "w", encoding="utf-8") as out_file:
    out_file.write(tabulate([r.values() for r in table], table[0].keys(), tablefmt='latex', floatfmt=".2e"))
with open(out_dir + "n_min_table_stop_d.csv", "w", encoding="utf-8") as out_file:
    out_file.write(tabulate([r.values() for r in table], table[0].keys(), tablefmt='tsv', floatfmt=".2e"))

