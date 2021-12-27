import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

import SVM
from SVR import SVR
from GVPM import GVPM
from experiments_ML.metrics import Scaler
from utils import plot_error
from metrics import mean_euclidean_error, min_max_scale

basedir =  ''
def load_cup_train():
    df = pd.read_csv(basedir + 'cup 2021/ML-CUP21-INT-TR.csv', header=None)
    x = df.iloc[:, 1:-2].to_numpy()
    y = df.iloc[:, -2:].to_numpy()
    return x, y



def load_cup_test():
    df = pd.read_csv(basedir+'cup 2021/ML-CUP21-TS.csv')
    x = df.iloc[:, 1:].to_numpy()
    return x

#
# print(np.max(load_cup_train()[1]), np.min(load_cup_train()[1]))
# print(np.std(load_cup_train()[1][:,1]), np.var(load_cup_train()[1]))
#
#
# X, y = load_cup_train()
#
#
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33)
# scaler = Scaler(0,1)
# X_train, scaled_y_train, X_valid, scaled_y_valid = scaler.scale(X_train, y_train, X_valid, y_valid)
#
# C = 200
# kernel = SVM.Kernels.RBF
# eps = 1e-3
#
# gamma = 'auto'
# tol = 1e-6
# degree = 3
# alpha_tol = 1e-3
# solver = GVPM(ls=GVPM.LineSearches.BACKTRACK, n_min=5, tol=tol, lam_low=1e-2, a_max=1e10,
#               a_min=1, plots=False, verbose=False, proj_tol=1e-8, max_iter=100000)
# # decomp_solver = GVPM(ls=GVPM.LineSearches.BACKTRACK, n_min=3, tol=1e-8, lam_low=1e-3, plots=False, verbose=False, proj_tol=1e-3)
# # solver = CplexSolver(tol = tol)
# my_model = SVR(solver = solver,
#             # exact_solver=CplexSolver(tol=tol, verbose=False),
#             C=C, kernel=kernel, eps=eps, gamma=gamma, degree=degree, alpha_tol=alpha_tol)
# train_err = []
# test_err = []
#
# sv = []
# sv_sota = []
#
# model_sota = MultiOutputRegressor(svm.SVR(C=C, kernel=kernel, epsilon=eps, gamma=gamma, degree=degree))
# train_err_sota = []
# test_err_sota = []
# batches = 10
# batch_size = int(len(X_train) / batches)
#
# print(batch_size)
# print(X_train.shape, y_train.shape)
#
#
# def train_and_test(model, test_err, train_err):
#     model.fit(X_train[:bs], scaled_y_train[:bs])
#
#     y_pred = model.predict(X_train)
#     y_pred = scaler.scale_back(y_pred)
#     train_err.append(mean_euclidean_error(y_pred, y_train))
#
#     y_pred = model.predict(X_valid)
#     y_pred = scaler.scale_back(y_pred)
#     test_err.append(mean_euclidean_error(y_pred, y_valid))
#     print(train_err)
#     print(test_err, "\n")
#
#
# for i in range(batches):
#     print("i={}----------------------------------------------------------------------".format(i))
#     bs = (i + 1) * batch_size
#
#     train_and_test(my_model, test_err, train_err)
#     train_and_test(model_sota, test_err_sota, train_err_sota)
#
# plot_error(train_err, test_err, "mySVR {} C={} eps={} tol={} solver={}".format(kernel, C, eps, tol, solver))
# plot_error(train_err_sota, test_err_sota, "sklearn {} C={} eps={}".format(kernel, C, eps))
# print(test_err)
# plot_sv_number(sv)
# plot_sv_number(sv_sota)
