import numpy as np
from joblib import Parallel, delayed
from sklearn import preprocessing
from tabulate import tabulate

import SVM
from SVR import SVR
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, euclidean_distances

from experiments_ML.metrics import mean_euclidean_error, mean_squared_error
from experiments_ML.Scaler import Scaler
from GVPM import GVPM
from Cplex_Solver import CplexSolver
from experiments_ML.load_cup_ds import load_cup_int_train
from sklearn.model_selection import KFold

X, y = load_cup_int_train()
print(X.shape, y.shape)

kf = KFold(n_splits=5)
def experiment(C, alpha_tol, eps, kernel, gamma, degree, tol):
    train_mse = []
    test_mse = []
    train_mee = []
    test_mee = []

    for train_ind, test_ind in kf.split(X):

        X_train = X[train_ind, :]
        y_train = y[train_ind, :]
        X_test = X[test_ind, :]
        y_test = y[test_ind, :]
        scaler = Scaler()
        # print(train_ind, test_ind, X_train.shape, y_test.shape)
        X_train, scaled_y_train, X_test, scaled_y_test = scaler.scale(X_train, y_train, X_test, y_test)

        # solver = GVPM(ls=GVPM.LineSearches.BACKTRACK, n_min=2, tol=tol, lam_low=1e-2, a_max=1e10,
        #               a_min=1, plots=False, verbose=False, proj_tol=1e-7, max_iter=1e5)
        solver = CplexSolver(tol = tol)
        model = SVR(solver=solver, C=C, kernel=kernel, eps=eps, gamma=gamma, degree=degree, alpha_tol=alpha_tol)
        try:
            model.fit(X_train, scaled_y_train)
            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)

            pred_train = scaler.scale_back(pred_train)
            pred_test = scaler.scale_back(pred_test)

            train_mse.append(mean_squared_error(y_train, pred_train))
            test_mse.append(mean_squared_error(y_test, pred_test))
            train_mee.append(mean_euclidean_error(y_train, pred_train))
            test_mee.append(mean_euclidean_error(y_test, pred_test))
            print(test_mee, train_mee)
        except Exception:
            print(kernel, C, alpha_tol, eps, gamma, degree, tol, Exception.__name__)

            return kernel, C, alpha_tol, eps, gamma, degree, tol, 0, 0, 0, Exception.__name__

    # print(train_err, test_err, train_mae, test_mae)
    train_mse = remove_nans(train_mse)
    test_mse = remove_nans(test_mse)
    train_mee = remove_nans(train_mee)
    test_mee = remove_nans(test_mee)

    r = kernel, C, alpha_tol, eps, gamma, degree, tol, \
        np.mean(train_mse, ), np.std(train_mse), np.mean(test_mse), np.std(test_mse), \
        np.mean(train_mee), np.std(train_mee), np.mean(test_mee), np.std(test_mee)
    with open("cup 2021/gs_temp/temp_poly.csv", "a", encoding="utf-8") as out_file:
        out_file.write(tabulate([r], tablefmt='tsv'))
        out_file.write("\n")
    print(kernel, C, alpha_tol, eps, gamma, degree, tol, np.mean(train_mee), np.std(train_mee), np.mean(test_mee), np.std(test_mee))

    return r


def remove_nans(train_mse):
    train_mse = np.array(train_mse)
    train_mse = train_mse[np.logical_not(np.isnan(train_mse))]
    return train_mse


kernels = [(k, g, d) for k in [SVM.Kernels.RBF, SVM.Kernels.Sigmoidal]
           for g in [SVM.Kernels.GammaModes.AUTO]
           for d in [1]]
poly_kernels = [(k, g, d) for k in [SVM.Kernels.POLY]
                for g in SVM.Kernels.GammaModes.ALL
                for d in [1, 3, 5, 7]]
kernels.extend(poly_kernels)
print(kernels)
# 6 jobs should be a good choice because the problem is subsolved by 2 coworkers
table = Parallel(n_jobs=6, verbose=11)(delayed(experiment)(C, alpha_tol, eps, kernel, gamma, degree, tol)
                                       for C in [5120,
                                            2560,
                                           1280,
                                                 640,
                                           320,
                                           160,80,40,20]
                                       for alpha_tol in [1e-1, 1e-3, 1e-5, 1e-7]
                                       for tol in [1e-3
                                           , 1e-5, 1e-7
                                                   ]
                                       for eps in [1e-2, 1e-3, 1e-1]
                                       for kernel, gamma, degree in poly_kernels
                                       )

header = ['Kernel', 'C', 'alpha tol', 'eps', 'gamma', 'degree',  'opt tol', 'train avg mse', 'train std mse',
          'test avg mse', 'test std mse', 'train avg mee', 'train std mee', 'test avg mee', 'test std mee']
with open("results/kcup_gs_results_poly.txt", "w", encoding="utf-8") as out_file:
    out_file.write(tabulate([r for r in table], header, tablefmt='latex'))
with open("results/kcup_gs_results_poly.csv", "w", encoding="utf-8") as out_file:
    out_file.write(tabulate([r for r in table], header, tablefmt='tsv'))
