import numpy as np
from joblib import Parallel, delayed
from sklearn import preprocessing
from tabulate import tabulate

import SVM
from SVR import SVR
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, euclidean_distances

from experiments_ML.metrics import Scaler, mean_euclidean_error, mean_squared_error
from GVPM import GVPM
from experiments_ML.load_cup_ds import load_cup_train
from sklearn.model_selection import KFold

X, y = load_cup_train()
print(X.shape, y.shape)


def experiment(C, alpha_tol, eps, kernel, gamma, degree, tol):
    train_mse = []
    test_mse = []
    train_mee = []
    test_mee = []
    kf = KFold(n_splits=5)
    for train_ind, test_ind in kf.split(X):

        X_train = X[train_ind, :]
        y_train = y[train_ind, :]
        X_test = X[test_ind, :]
        y_test = y[test_ind, :]
        scaler = Scaler()

        X_train, scaled_y_train, X_valid, scaled_y_test = scaler.scale(X_train, y_train, X_test, y_test)

        solver = GVPM(ls=GVPM.LineSearches.BACKTRACK, n_min=3, tol=tol, lam_low=1e-3, plots=False, proj_tol=1e-3)
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
        except Exception:
            return kernel, C, alpha_tol, eps, gamma, degree, tol, 0, 0, 0, Exception.__name__

    # print(train_err, test_err, train_mae, test_mae)

    r = kernel, C, alpha_tol, eps, gamma, degree, tol, np.mean(train_mse), np.std(train_mse), np.mean(test_mse), np.std(
        test_mse), np.mean(train_mee), np.std(train_mee), np.mean(test_mee), np.std(test_mee)
    with open("cup 2021/gs_temp/temp.csv", "a", encoding="utf-8") as out_file:
        out_file.write(tabulate([r], tablefmt='tsv'))
        out_file.write("\n")
    return r


kernels = [(k, g, d) for k in [SVM.Kernels.RBF, SVM.Kernels.Sigmoidal]
           for g in SVM.Kernels.GammaModes.ALL
           for d in [1]]
poly_kernels = [(k, g, d) for k in [SVM.Kernels.RBF, SVM.Kernels.POLY]
                for g in SVM.Kernels.GammaModes.ALL
                for d in [1, 3, 5, 7]]
# kernels.extend(poly_kernels)
print(kernels)

table = Parallel(n_jobs=-3, verbose=11)(delayed(experiment)(C, alpha_tol, eps, kernel, gamma, degree, tol)
                                       for C in [1, 10, 20, 40, 80, 160, 320, 640, 1280]
                                       for alpha_tol in [1e-1, 1e-2, 1e-3, 1e-5]
                                       for tol in [1e-3, 1e-5, 1e-7]
                                       for eps in [1e-1, 1e-2, 1e-3]
                                       for kernel, gamma, degree in kernels
                                       )

header = ['Kernel', 'C', 'alpha tol', 'eps', 'gamma', 'degree',  'opt tol', 'train avg mse', 'train std mse',
          'test avg mse', 'test std mse', 'train avg mee', 'train std mee', 'test avg mee', 'test std mee']
with open("results/kcup_gs_results.txt", "w", encoding="utf-8") as out_file:
    out_file.write(tabulate([r for r in table], header, tablefmt='latex'))
with open("results/kcup_gs_results.csv", "w", encoding="utf-8") as out_file:
    out_file.write(tabulate([r for r in table], header, tablefmt='tsv'))
