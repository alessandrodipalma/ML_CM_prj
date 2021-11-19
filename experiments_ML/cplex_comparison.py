from joblib import Parallel, delayed
from sklearn import preprocessing
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from GVPM import GVPM

from experiments_ML.svr_experiment import experiment

header = ['monk', 'requested accuracy', 'cplex_acc', 'gvpm_acc', 'cplex_time', 'cplex_tsa', 'gvpm_tsa', 'gvpm_config']

kernels = [
    ('rbf', 'auto', 1),
    #        ('rbf', 'scale', 1),
    #        ('poly', 'scale', 3),
    #        ('poly', 'scale', 5),
    # ('poly', 'scale', 7),
    # ('linear', 'scale', 1)
]





# for a in ds:
#     print(a.shape)
# table = []
# for C in [10, 50, 100]:
#     for eps in [1e-2, 1e-3]:
#         for tol in [1e-1, 1e-3]:
#             for kernel, gamma, degree in kernels:
#                 for ls in GVPM.LineSearches.values:
#                     for a_min in [1e-1, 1e-2, 1e-4, 1e-8]:
#                         for a_max in [10, 1e2, 1e4, ]:
#                             for n_min in [1, 2, 4, 8]:
#                                 for lam_low in [1e-1, 1e-2, 1e-3]:
#                                     for stopping_rule in GVPM.StoppingRules.values:
#                                         for proj_tol in [1e-1, 1e-2, 1e-4]:
#                                             table.append(experiment(ds, kernel, gamma, degree, C, eps, tol,
#                                                                     ls=ls, a_min=a_min, a_max=a_max, n_min=n_min, lam_low=lam_low,
#                                                                     stopping_rule=stopping_rule,
#                                                                     proj_tol=proj_tol
#                                                                     ))

feature_samples_dict = [
                        {'features': 10, 'samples': 200},
                        {'features': 50, 'samples': 200},
                        {'features': 100, 'samples': 200},
                        {'features': 200, 'samples': 200},
                        {'features': 400, 'samples': 200},
                        # {'features': 200, 'samples': 500},
                        # {'features': 300, 'samples': 1000},
                        ]


for d in feature_samples_dict:
    features, samples = d.values()
    X, y = make_regression(n_samples=samples, n_features=features, random_state=42)
    X = preprocessing.StandardScaler().fit(X).transform(X)
    y = 2 * (y - min(y)) / (max(y) - min(y)) - 1

    ds = train_test_split(X, y, test_size=0.3, random_state=42)

    table = Parallel(n_jobs=12)(delayed(experiment)(ds, kernel, gamma, degree, C, eps, tol=tol,
                                                    ls=ls, a_min=a_min, a_max=a_max, n_min=n_min, lam_low=lam_low,
                                                    stopping_rule=stopping_rule,
                                                    proj_tol=proj_tol
                                                    )
                                for C in [1]
                                for eps in [1e-2]
                                for tol in [1e-1, 1e-2, 1e-4, 1e-6]
                                for kernel, gamma, degree in kernels
                                for ls in GVPM.LineSearches.values
                                for a_min in [1, 1e-2, 1e-4, 1e-8]
                                for a_max in [1, 1e2, 1e4, 1e8]
                                for n_min in [2, 4, 8]
                                for lam_low in [1e-1, 1e-2, 1e-3]
                                for stopping_rule in [GVPM.StoppingRules.gradient]
                                for proj_tol in [1e-1, 1e-2, 1e-4, 1e-6]
                                )

    header = ["kernel", "gamma", "degree", "C", "tol",
              "cp_train_mse", "cp_test_mse", "cp_train_mae", "cp_test_mae", "cp_time", "cp_gap", "cp_x", "cp_iter",
              "gvpm_train_mse", "gvpm_test_mse", "gvpm_train_mae", "gvpm_test_mae", "gvpm_time", "gvpm_gap", "gvpm_x", "gvpm_iter",
              "ls", "a_min", "a_max", "n_min", "lam_low", "stopping_rule", "proj_tol"]

    with open("cplex_results/gs_results_{}_{}.csv".format(features, samples), "w", encoding="utf-8") as out_file:
        out_file.write(tabulate([r for r in table], header, tablefmt='tsv'))
