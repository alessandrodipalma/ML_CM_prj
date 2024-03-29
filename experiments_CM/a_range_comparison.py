"""
How the range [a_min, a_max] affects convergence
ranges: +- 1e-2 , -+ 1e-6 , +- 1e-12
"""
import os

from utils import generate_regression_from_feature_sample_dict
import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate

from Cplex_Solver import CplexSolver
from SVR import SVR
from GVPM import GVPM


feature_samples_dict = [{'features': 10, 'samples': 200},
                        {'features': 50, 'samples': 200},
                        {'features': 100, 'samples': 100},
                        {'features': 100, 'samples': 500},
                        {'features': 200, 'samples': 500},
                        {'features': 300, 'samples': 1000},
                        ]

out_dir = "plots/a_upp/big_comp_lamupp1/"
os.mkdir(out_dir)
a_ranges = [1, 1e-1, 1e-2, 1e-4, 1e-8]

n_problems = 1
all_problems = generate_regression_from_feature_sample_dict(feature_samples_dict, n_problems)

combs = [
    (GVPM.LineSearches.EXACT, 1),
    (GVPM.LineSearches.EXACT, 1e30),
    (GVPM.LineSearches.BACKTRACK, 1),
    (GVPM.LineSearches.BACKTRACK, 1e30)
]

for i, (ls, amax) in enumerate(combs):
    histories = []
    table = []
    plt.rcParams["figure.figsize"] = (20, 20)
    plt.title("Lambda Variation")
    cols = 2
    rows = 3
    fig, axs = plt.subplots(rows, cols)
    plt.yscale('log')

    for i, d in enumerate(feature_samples_dict):
        C = 1
        kernel = 'rbf'
        eps = 0.1
        gamma = 'scale'
        tol = 1e-2
        # ls = GVPM.LineSearches.BACKTRACK

        row = {'features': d['features'], 'samples': d['samples']}
        histories = {}

        plot = axs[int(i / cols), i % cols]

        for t, a in enumerate(a_ranges):
            amin = a
            # amax = 1e8
            solver = GVPM(ls=ls, n_min=2, tol=tol, lam_low=1e-3, plots=False, proj_tol=1e-2, a_min=amin,
                          a_max=amax,
                          lam_upp= 1
                          )
            stats = []

            for p in all_problems[i]:
                X, y = p

                model = SVR(solver=solver, C=C, kernel=kernel, eps=eps, gamma=gamma,
                            exact_solver=CplexSolver(tol=1e-10, verbose=False))
                n_sv, alphas, indices = model.fit(X, y)
                stats.append(solver.stats)

            it = 0
            f_gap = 0
            time_spent = 0
            for s in stats:
                it += s['it']
                f_gap += solver.f_gap_history[-1]
                time_spent += s['time_tot']
            # final_stats =
            row["{} {} it".format(ls, a)] = it / n_problems
            row["{} {} f_gap".format(ls, a)] = f_gap / n_problems
            row["{} {} time".format(ls, a)] = time_spent / n_problems


            plot.plot(np.arange(len(solver.f_gap_history)), solver.f_gap_history, label='[{},{}]'.format(amin, amax))
        # plot.axhline(y=1, color='r', linestyle='-')
        plot.set_title("{} features, {} samples".format(d["features"], d["samples"]))
        plot.legend()
        plot.set_yscale('log')

        table.append(row)

    # plt.show()
    plt.savefig(out_dir + "a_range_{}_{}.png".format(ls, amax))

    with open(out_dir + "table_a_range_{}_{}.txt".format(ls, amax), "w", encoding="utf-8") as out_file:
        out_file.write(tabulate([r.values() for r in table], table[0].keys(), tablefmt='latex', floatfmt=".2e"))
    with open(out_dir + "table_a_range_{}_{}.csv".format(ls, amax), "w", encoding="utf-8") as out_file:
        out_file.write(tabulate([r.values() for r in table], table[0].keys(), tablefmt='csv', floatfmt=".2e"))


