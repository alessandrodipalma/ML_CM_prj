import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

import SVM
from Cplex_Solver import CplexSolver
from GVPM import GVPM
from SVR import SVR
from experiments_ML.load_cup_ds import load_cup_train
from experiments_ML.metrics import mean_euclidean_error, Scaler
from utils import plot_error, plot_sv_number

print(np.max(load_cup_train()[1]), np.min(load_cup_train()[1]))
print(np.std(load_cup_train()[1][:,1]), np.var(load_cup_train()[1]))


X, y = load_cup_train()


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
scaler = Scaler(0,1)
X_train, scaled_y_train, X_valid, scaled_y_valid = scaler.scale(X_train, y_train, X_valid, y_valid)

C = 2560
kernel = SVM.Kernels.RBF
eps = 1e-3

gamma = SVM.Kernels.GammaModes.SCALE
tol = 1e-3
degree = 3
alpha_tol = 1e-3
# solver = GVPM(ls=GVPM.LineSearches.BACKTRACK, n_min=5, tol=tol, lam_low=1e-2, a_max=1e10,
#               a_min=1, plots=False, verbose=False, proj_tol=1e-8, max_iter=1e5)

solver = GVPM(ls=GVPM.LineSearches.BACKTRACK, n_min=2, tol=tol, lam_low=1e-2, a_max=1e10,
                      a_min=1, plots=False, verbose=False, proj_tol=1e-7, max_iter=1e4)
# decomp_solver = GVPM(ls=GVPM.LineSearches.BACKTRACK, n_min=3, tol=1e-8, lam_low=1e-3, plots=False, verbose=False, proj_tol=1e-3)
solver = CplexSolver(tol = tol)
my_model = SVR(solver = solver,
            # exact_solver=CplexSolver(tol=tol, verbose=False),
            C=C, kernel=kernel, eps=eps, gamma=gamma, degree=degree, alpha_tol=alpha_tol)
train_err = []
test_err = []

sv = []
sv_sota = []

model_sota = MultiOutputRegressor(svm.SVR(C=C, kernel=kernel, epsilon=eps, gamma=gamma, degree=degree))
train_err_sota = []
test_err_sota = []
batches = 1
batch_size = int(len(X_train) / batches)

print(batch_size)
print(X_train.shape, y_train.shape)


def train_and_test(model, test_err, train_err):
    model.fit(X_train[:bs], scaled_y_train[:bs])

    y_pred = model.predict(X_train)
    y_pred = scaler.scale_back(y_pred)
    train_err.append(mean_euclidean_error(y_pred, y_train))

    y_pred = model.predict(X_valid)
    y_pred = scaler.scale_back(y_pred)
    test_err.append(mean_euclidean_error(y_pred, y_valid))
    print(train_err)
    print(test_err, "\n")


for i in range(batches):
    print("i={}----------------------------------------------------------------------".format(i))
    bs = (i + 1) * batch_size

    train_and_test(my_model, test_err, train_err)
    train_and_test(model_sota, test_err_sota, train_err_sota)

plot_error(train_err, test_err, "mySVR {} C={} eps={} tol={} solver={}".format(kernel, C, eps, tol, solver))
plot_error(train_err_sota, test_err_sota, "sklearn {} C={} eps={}".format(kernel, C, eps))
print(test_err)
# plot_sv_number(sv)
# plot_sv_number(sv_sota)