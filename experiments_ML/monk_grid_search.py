import sys

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

import SVM
from GVPM import GVPM
from SVM import SVM as SVM_class
from experiments_ML.load_monk import load_monk
from experiments_ML.Scaler import Scaler


def experiment(monk, C, alpha_tol, kernel, gamma, degree, tol):
    X_train, y_train = load_monk(monk, 'train')
    X_test, y_test = load_monk(monk, 'test')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = np.where(y_train == 0, -1, y_train)

    solver = GVPM(ls=GVPM.LineSearches.BACKTRACK, n_min=2, tol=tol, lam_low=1e-2, proj_tol=1e-8, a_min = 1e-1, a_max=1e10, plots=False, max_iter=1e5)
    model = SVM_class(solver=solver,C=C, kernel=kernel, gamma=gamma, degree=degree, alpha_tol=alpha_tol, verbose=False)


    try:
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        pred_train = np.where(pred_train == -1, 0, pred_train)
        pred_test = np.where(pred_test == -1, 0, pred_test)
        y_train = np.where(y_train == -1, 0, y_train)

        train_mse = mean_squared_error(y_train, pred_train)
        test_mse = mean_squared_error(y_test, pred_test)
        train_acc = accuracy_score(y_train, pred_train)
        test_acc = accuracy_score(y_test, pred_test)
    except Exception as exc:
        print(sys.exc_info()[2])
        print(exc)
        return kernel, C, alpha_tol, gamma, degree, tol, 0, 0, 0, exc


    r = kernel, C, alpha_tol, gamma, degree, tol, train_mse, test_mse, train_acc, test_acc
    with open("results_monks/monk{}_temp.csv".format(monk), "a", encoding="utf-8") as out_file:
        out_file.write(tabulate([r], tablefmt='tsv'))
        out_file.write("\n")
    return r



kernels = [(k, g, d) for k in [SVM.Kernels.RBF, SVM.Kernels.Sigmoidal]
           for g in [SVM.Kernels.GammaModes.AUTO]
           for d in [1]]
poly_kernels = [(k, g, d) for k in [SVM.Kernels.POLY]
                for g in [SVM.Kernels.GammaModes.AUTO]
                for d in [
                    # 1,2,3,4,5,6,
                    # 9,10,11,12,
                    13,17,21,24,25,
                    27,39,
                    100
                ]]

# kernels.extend(poly_kernels)
for monk in [2]:
    table = Parallel(n_jobs=8, verbose=11)(delayed(experiment)(monk, C, alpha_tol, kernel, gamma, degree, tol)
                                  for C in [
                                               # 0.5,
                                               # 1,
                                               # 5,
                                               # 10,
                                               # 50,
                                               # 100,
                                               # 200,
                                               # 400,
                                               # 800,
                                               # 1600,
                                               # 3200
        6400,
                                               12800,
        #                                        25600
                                           ]
                                for alpha_tol in [
                                               1e-1,
                                               1e-3,
                                               1e-5,
                                               #    1e-7
                                                  ]
                                  for tol in [
                                               1e-1,
                                              1e-3,
                                              1e-5, 1e-7,
                                              # 1e-9
                                              ]
                                  for kernel, gamma, degree in poly_kernels)

    header = ['Monk', 'Kernel', 'alpha_tol', 'Gamma', 'Degree', 'C', 'opt tol', 'train mse', 'test mse', 'train acc', 'test acc']
    with open("monk_{}_gs2_results.txt".format(monk), "w", encoding="utf-8") as out_file:
        out_file.write(tabulate([r for r in table], header, tablefmt='latex'))
    with open("monk_{}_gs2_results.csv".format(monk), "w", encoding="utf-8") as out_file:
        out_file.write(tabulate([r for r in table], header, tablefmt='tsv'))