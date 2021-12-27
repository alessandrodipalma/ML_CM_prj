import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

from GVPM import GVPM
from experiments_ML.load_monk import load_monk
from SVM import SVM
from sklearn.metrics import mean_squared_error as mse, accuracy_score


def experiment(monk_n, kernel, gamma, degree, C, tol, data_used=1):
    X_train, y_train = load_monk(monk_n, 'train')
    X_test, y_test = load_monk(monk_n, 'test')
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if data_used < 1:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=data_used, random_state=42)

    y_train = np.where(y_train == 0, -1, y_train)


    solver = GVPM(ls=GVPM.LineSearches.BACKTRACK, n_min=2, tol=tol, lam_low=1e-3, proj_tol=1e-3)
    model = SVM(solver=solver,
                # exact_solver=CplexSolver(tol=tol, verbose=False),
                C=C, kernel=kernel, gamma=gamma, degree=degree)

    try:
        n_sv, alphas, indices = model.fit(X_train, y_train)
    except ZeroDivisionError:
        return monk_n, kernel, gamma, degree, C, tol, "Zero Division Error", 0,0,0
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_pred = np.where(train_pred == -1, 0, train_pred)
    test_pred = np.where(test_pred == -1, 0, test_pred)
    y_train = np.where(y_train == -1, 0, y_train)

    train_acc = accuracy_score(train_pred, y_train)
    train_mse = mse(train_pred, y_train)
    test_acc = accuracy_score(test_pred, y_test)
    test_mse = mse(test_pred, y_test)

    return monk_n, kernel, gamma, degree, C, tol, train_mse, test_mse, train_acc, test_acc


kernels = [('rbf', 'auto', 1),
           ('rbf', 'scale', 1),
           ('poly', 'scale', 3),
           ('poly', 'scale', 5),
           ('poly', 'scale', 7),
           ('linear', 'scale', 1)]

for monk in [1, 2, 3]:
    table = Parallel(n_jobs=12)(delayed(experiment)(monk, kernel, gamma, degree, C, tol)
                                  for C in [0.1, 0.5, 1, 5, 10, 50, 100]
                                  for tol in [1e-1, 1e-3, 1e-8]
                                  for kernel, gamma, degree in kernels)

    header = ['Monk', 'Kernel', 'Gamma', 'Degree', 'C', 'opt tol', 'train mse', 'test mse', 'train acc', 'test acc']
    with open("monk_{}_gs_results.txt".format(monk), "w", encoding="utf-8") as out_file:
        out_file.write(tabulate([r for r in table], header, tablefmt='latex'))
    with open("monk_{}_gs_results.csv".format(monk), "w", encoding="utf-8") as out_file:
        out_file.write(tabulate([r for r in table], header, tablefmt='tsv'))