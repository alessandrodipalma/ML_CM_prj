import time

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error as mse, mean_absolute_error as mae
from tabulate import tabulate

from Cplex_Solver import CplexSolver
from GVPM import GVPM
from SVR import SVR
from numpy.linalg import norm


def experiment(ds, kernel, gamma, degree, C, eps, tol,
               ls, a_min, a_max, n_min, lam_low,
               stopping_rule,
               proj_tol,
               data_used=1):
    exact_solver = CplexSolver(tol=1e-8, verbose=False)
    cplex_solver = CplexSolver(tol=tol, verbose=False)



    gvpm_solver = GVPM(ls=ls, a_min=a_min, a_max=a_max, n_min=n_min, tol=tol, lam_low=lam_low,
                       stopping_rule=stopping_rule, proj_tol=proj_tol)

    def solve(solver, X_train, X_test, y_train, y_test):
        model = SVR(solver=solver, C=C, kernel=kernel, eps=eps, gamma=gamma, degree=degree)

        try:
            n_sv, alphas, indices = model.train(X_train, y_train)
        except ZeroDivisionError:
            return 0, 0, 0, 0, 0
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_pred = np.where(train_pred == -1, 0, train_pred)
        test_pred = np.where(test_pred == -1, 0, test_pred)
        y_train = np.where(y_train == -1, 0, y_train)

        train_mae = mae(train_pred, y_train)
        train_mse = mse(train_pred, y_train)
        test_mae = mae(test_pred, y_test)
        test_mse = mse(test_pred, y_test)

        return train_mae, train_mse, test_mae, test_mse, solver.elapsed_time, solver.iterations

    exact_train_mse, exact_test_mse, exact_train_acc, cp_test_acc, exact_time, exact_iter = solve(exact_solver, *ds)
    cp_train_mse, cp_test_mse, cp_train_acc, cp_test_acc, cp_time, cp_iter = solve(cplex_solver, *ds)
    gvpm_train_mse, gvpm_test_mse, gvpm_train_acc, gvpm_test_acc, gvpm_time, gvpm_iter = solve(gvpm_solver, *ds)

    cp_gap = abs(cplex_solver.f_value - exact_solver.f_value)
    gvpm_gap = abs(gvpm_solver.f_value - exact_solver.f_value)
    cp_x = norm(cplex_solver.x_value)
    gvpm_x = norm(gvpm_solver.x_value)
    # table = []
    # table.append([kernel, gamma, degree, C, tol,
    #        cp_train_mse, cp_test_mse, cp_train_acc, cp_test_acc, cp_time,
    #        gvpm_train_mse, gvpm_test_mse, gvpm_train_acc, gvpm_test_acc, gvpm_time,
    #        ls, a_min, a_max, n_min, lam_low, stopping_rule, proj_tol])
    # with open("cplex_results/partial_results.csv", "a", encoding="utf-8") as out_file:
    #     out_file.write(tabulate([r for r in table], tablefmt='tsv'))

    return kernel, gamma, degree, C, tol, \
           cp_train_mse, cp_test_mse, cp_train_acc, cp_test_acc, cp_time, cp_gap, cp_x, cp_iter, \
           gvpm_train_mse, gvpm_test_mse, gvpm_train_acc, gvpm_test_acc, gvpm_time, gvpm_gap, gvpm_x, gvpm_iter, \
           ls, a_min, a_max, n_min, lam_low, stopping_rule, proj_tol
