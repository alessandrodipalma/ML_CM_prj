import numpy as np
from joblib import Parallel, delayed
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from SVR import SVR
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from gvpm import GVPM
from load_cup_ds import load_cup_train

X, y = load_cup_train()
X = preprocessing.StandardScaler().fit(X).transform(X)

n_split = 10


def experiment(C, eps, kernel, gamma, degree, tol, split):
    train_err = []
    test_err = []
    train_mae = []
    test_mae = []
    for i in range(n_split):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)

        y_scaler = preprocessing.StandardScaler().fit(y_train)
        y_train = y_scaler.transform(y_train)
        y_test = y_scaler.transform(y_test)

        solver = GVPM(ls=GVPM.LineSearches.BACKTRACK, n_min=3, tol=tol, lam_low=1e-3, plots=False, proj_tol=1e-3)
        model = SVR(solver=solver, C=C, kernel=kernel, eps=eps, gamma=gamma, degree=degree)
        try:
            model.train(X_train, y_train)
            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)
            # rev_train_pred = y_scaler.inverse_transform(pred_train)
            # rev_test_pred = y_scaler.inverse_transform(pred_test)

            train_err.append(mse(y_train, pred_train))
            test_err.append(mse(y_test, pred_test))
            train_mae.append(mae(y_train, pred_train))
            test_mae.append(mae(y_test, pred_test))
        except Exception:
            return kernel, C, eps, gamma, degree, tol, split, 0, 0, 0, Exception.__name__


    print(train_err, test_err, train_mae, test_mae)

    return kernel, C, eps, gamma, degree, tol, split, np.mean(train_err), np.std(train_err), np.mean(test_err), np.std(
        test_err), np.mean(train_mae), np.std(train_mae), np.mean(test_mae), np.std(test_mae)


kernels = [
    ('rbf', 'auto', 1),
    ('rbf', 'scale', 1),
    ('poly', 'scale', 3),
    ('poly', 'scale', 5),
    ('poly', 'scale', 7),
    ('linear', 'scale', 1)
]

table = Parallel(n_jobs=12)(delayed(experiment)(C, eps, kernel, gamma, degree, tol, split)
                           for C in [0.1, 0.5, 1, 5, 10, 50, 100]
                           for tol in [1e-1, 1e-3, 1e-8]
                           for eps in [1e-1, 1e-2, 1e-4, 1e-8]
                           for kernel, gamma, degree in kernels
                           for split in [0.5,0.33,0.2]
                           )

header = ['Kernel', 'C', 'eps', 'gamma', 'degree', 'opt tol', 'tr/val split', 'train avg mse', 'train std mse',
          'test avg mse', 'test std mse', 'train avg mae', 'train std mae', 'test avg mae', 'test std mae']
with open("cup_gs_resultsx10.txt", "w", encoding="utf-8") as out_file:
    out_file.write(tabulate([r for r in table], header, tablefmt='latex'))
with open("cup_gs_resultsx10.csv", "w", encoding="utf-8") as out_file:
    out_file.write(tabulate([r for r in table], header, tablefmt='tsv'))
