import numpy as np
from joblib import Parallel, delayed
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from SVR import SVR
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, euclidean_distances
from gvpm import GVPM
from load_cup_ds import load_cup_train
from sklearn.model_selection import KFold

X, y = load_cup_train()
X = preprocessing.StandardScaler().fit(X).transform(X)
print(X.shape, y.shape)


def experiment(C, eps, kernel, gamma, degree, tol):
    train_err = []
    test_err = []
    train_mae = []
    test_mae = []
    train_mee = []
    test_mee = []
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):

        X_train = X[train, :]
        y_train = y[train, :]
        X_test = X[test, :]
        y_test = y[test, :]
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        y_scaler = preprocessing.StandardScaler().fit(y_train)
        y_train = y_scaler.transform(y_train)
        y_test = y_scaler.transform(y_test)

        solver = GVPM(ls=GVPM.LineSearches.BACKTRACK, n_min=3, tol=tol, lam_low=1e-3, plots=False, proj_tol=1e-3)
        model = SVR(solver=solver, C=C, kernel=kernel, eps=eps, gamma=gamma, degree=degree)
        try:
            model.train(X_train, y_train)
            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)

            train_err.append(mse(y_train, pred_train))
            test_err.append(mse(y_test, pred_test))
            train_mae.append(mae(y_train, pred_train))
            test_mae.append(mae(y_test, pred_test))
            train_mee.append(np.mean(euclidean_distances(y_train, pred_train)))
            test_mee.append(np.mean(euclidean_distances(y_test, pred_test)))
        except Exception:
            return kernel, C, eps, gamma, degree, tol, 0, 0, 0, Exception.__name__

    print(train_err, test_err, train_mae, test_mae)

    return kernel, C, eps, gamma, degree, tol, np.mean(train_err), np.std(train_err), np.mean(test_err), np.std(
        test_err), np.mean(train_mae), np.std(train_mae), np.mean(test_mae), np.std(test_mae), \
           np.mean(train_mee), np.std(test_mee), np.mean(test_mee), np.std(test_mee)


kernels = [
    ('rbf', 'auto', 1),
    ('rbf', 'scale', 1),
    ('poly', 'scale', 3),
    # ('poly', 'scale', 5),
    ('poly', 'scale', 7),
    ('linear', 'scale', 1)
]

table = Parallel(n_jobs=4)(delayed(experiment)(C, eps, kernel, gamma, degree, tol)
                           for C in [1,
                                     10, 100
                                     ]
                           for tol in [1e-1,
                                       1e-3
                                       ]
                           for eps in [1e-1,
                                       1e-2, 1e-3
                                       ]
                           for kernel, gamma, degree in kernels
                           )

header = ['Kernel', 'C', 'eps', 'gamma', 'degree', 'opt tol', 'train avg mse', 'train std mse',
          'test avg mse', 'test std mse', 'train avg mae', 'train std mae', 'test avg mae', 'test std mae',
          'train avg mee', 'train std mee', 'test avg mee', 'test std mee']
with open("kcup_gs_resultsx10.txt", "w", encoding="utf-8") as out_file:
    out_file.write(tabulate([r for r in table], header, tablefmt='latex'))
with open("kcup_gs_resultsx10.csv", "w", encoding="utf-8") as out_file:
    out_file.write(tabulate([r for r in table], header, tablefmt='tsv'))
