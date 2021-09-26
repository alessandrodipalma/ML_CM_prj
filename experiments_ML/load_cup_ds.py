import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.metrics.pairwise import paired_distances
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error as mse, euclidean_distances
from SVR import SVR
from GVPM import GVPM
from utils import plot_error

basedir =  ''
def load_cup_train():
    df = pd.read_csv(basedir + 'monk/ML-CUP20-TR.csv')
    x = df.iloc[:, 1:].to_numpy()
    y = df.iloc[:, :2].to_numpy()
    return x, y


def load_cup_test():
    df = pd.read_csv(basedir+'monk/ML-CUP20-TS.csv')
    x = df.iloc[:, 1:].to_numpy()
    return x


print(np.max(load_cup_train()[1]), np.min(load_cup_train()[1]))
print(np.std(load_cup_train()[1][:,1]), np.var(load_cup_train()[1]))

#
# X, y = load_cup_train()
# X = preprocessing.StandardScaler().fit(X,y).transform(X)
# y_scaler = preprocessing.StandardScaler()
# y = y_scaler.fit(y).transform(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
#
# C = 10
# kernel = 'rbf'
# eps = 0.1
# gamma = 'auto'
# tol = 1e-10
#
# solver = GVPM(ls=GVPM.LineSearches.BACKTRACK, n_min=3, tol=1e-3, lam_low=1e-3, plots=False, verbose=False, proj_tol=1e-3)
# decomp_solver = GVPM(ls=GVPM.LineSearches.BACKTRACK, n_min=3, tol=1e-8, lam_low=1e-3, plots=False, verbose=False, proj_tol=1e-3)
#
# model = SVR(solver = solver,
#             decomp_solver = decomp_solver,
#             # exact_solver=CplexSolver(tol=tol, verbose=False),
#             C=C, kernel=kernel, eps=eps, gamma=gamma, degree=4)
# train_err = []
# test_err = []
#
# sv = []
# sv_sota = []
#
# model_sota = MultiOutputRegressor(svm.SVR(C=C, kernel=kernel, epsilon=eps, gamma=gamma))
# train_err_sota = []
# test_err_sota = []
#
# batch_size = int(len(X_train) / 4)
#
# print(batch_size)
# print(X_train.shape, y_train.shape)
# for i in range(4):
#     print("i={}----------------------------------------------------------------------".format(i))
#     bs = (i + 1) * batch_size
#
#     model.train(X_train[:bs], y_train[:bs])
#     train_err.append(np.mean(paired_distances(model.predict(X_train), y_train)))
#     test_err.append(np.mean(paired_distances(model.predict(X_test), y_test)))
#     print(train_err)
#     model_sota.fit(X_train[:bs], y_train[:bs])
#     train_err_sota.append(np.mean(paired_distances(model_sota.predict(X_train), y_train)))
#     test_err_sota.append(np.mean(paired_distances(model_sota.predict(X_test), y_test)))
#
# plot_error(train_err, test_err, "mySVR {} C={} eps={} tol={} solver={}".format(kernel, C, eps, tol, solver))
# plot_error(train_err_sota, test_err_sota, "sklearn {} C={} eps={}".format(kernel, C, eps))
# print(test_err)
# plot_sv_number(sv)
# plot_sv_number(sv_sota)