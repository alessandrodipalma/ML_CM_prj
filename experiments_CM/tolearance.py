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
                        {'features': 50, 'samples': 200},
                        {'features': 100, 'samples': 200},
                        {'features': 200, 'samples': 200},
                        {'features': 400, 'samples': 200},
                        {'features': 100, 'samples': 500},
                        # {'features': 800, 'samples': 200},
                        # {'features': 100, 'samples': 200},

                        # {'features': 100, 'samples': 1000},
                        # {'features': 300, 'samples': 1000},
                        # {'features': 500, 'samples': 2000},
                        ]
out_dir = "./plots/tolerance/"
# os.mkdir("./plots")
# os.mkdir("./plots/n_min/")
tols = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

n_problems = 20
all_problems = generate_regression_from_feature_sample_dict(feature_samples_dict, n_problems, fixed_rs=42)
tols.reverse()
for stopping_rule in GVPM.StoppingRules.values:
    histories = []
    table = []
    plt.rcParams["figure.figsize"] = (10, 50)
    # plt.title("n_min Variation")
    cols = 1
    rows = len(feature_samples_dict)
    fig, axs = plt.subplots(rows, cols)
    plt.yscale('log')

    for i, d in enumerate(feature_samples_dict):
        C = 1
        kernel = 'rbf'
        eps = 0.1
        gamma = 'scale'
        # tol = 1e-5
        ls = GVPM.LineSearches.BACKTRACK


        histories = {}

        plot = axs[i]

        row = {'features': d['features'], 'samples': d['samples']}
        k = 0
        it = {}
        f_gap = {}

        for tol in tols:
            it[i, tol] = 0
            f_gap[i, tol] = 0

        for p in all_problems[i]:

            solver = GVPM(ls=ls, n_min=2, tol=1e-8, lam_low=1e-3,
                          plots=False, proj_tol=1e-8, max_iter=2000, stopping_rule=stopping_rule, checkpointing=True)
            stats = []

            X, y = p

            model = SVR(solver=solver, C=C, kernel=kernel, eps=eps, gamma=gamma,
                        exact_solver=CplexSolver(tol=1e-10, verbose=False))
            n_sv, alphas, indices = model.train(X, y)
            stats.append(solver.stats)
            k += solver.cond

            for tol in tols.__reversed__():
                iterates, final_gap = solver.checkpoints[tol].values()
                it[i, tol] += iterates
                f_gap[i, tol] += final_gap

        row["cond"] = k / n_problems

        for t, tol in enumerate(tols):
            iterates, final_gap = solver.checkpoints[tol].values()
            plot.plot(np.arange(iterates), solver.f_gap_history[:iterates], label='tol = {}'.format(tol), linewidth=2)
            row["{} it".format(tol)] = it[i, tol] / n_problems
            row["{} f_gap".format(tol)] = f_gap[i, tol] / n_problems
        plot.set_title("{} features, {} samples, k={}".format(d['features'], d['samples'], solver.cond))
        plot.legend()
        plot.set_yscale('log')

        table.append(row)

    # plt.show()
    plt.savefig(out_dir + "{}_tol.png".format(stopping_rule))
    with open(out_dir + "table_{}.txt".format(stopping_rule), "w", encoding="utf-8") as out_file:
        out_file.write(tabulate([r.values() for r in table], table[0].keys(), tablefmt='latex', floatfmt=".2e"))

