import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

import Cplex_Solver
from GVPM import GVPM
from experiments_ML.load_monk import load_monk
from SVM import SVM
from sklearn.metrics import mean_squared_error as mse, accuracy_score


def experiment(monk_n, kernel, gamma, degree, C, tol, index=""):
    X_train_all, y_train_all = load_monk(monk_n, 'train')
    X_test, y_test = load_monk(monk_n, 'test')

    scaler = StandardScaler()
    X_train_all = scaler.fit_transform(X_train_all)
    X_test = scaler.transform(X_test)
    train_acc = []
    test_acc = []
    train_mse = []
    test_mse = []

    epochs = 10
    for k in range(epochs):

        bs = int(len(X_train_all)/epochs) * (k+1)
        print("samples", bs)
        X_train = X_train_all[:bs]
        y_train = y_train_all[:bs]
        print(X_train.shape, y_train.shape)
        y_train = np.where(y_train == 0, -1, y_train)


        # solver = GVPM(ls=GVPM.LineSearches.BACKTRACK, n_min=2, tol=tol, lam_low=1e-3, proj_tol=1e-3)
        solver= Cplex_Solver.CplexSolver(tol=tol)
        model = SVM(solver=solver,
                    # exact_solver=CplexSolver(tol=tol, verbose=False),
                    C=C, kernel=kernel, gamma=gamma, degree=degree)

        try:
            n_sv, alphas, indices = model.fit(X_train, y_train)
        except ZeroDivisionError:
            return monk_n, kernel, gamma, degree, C, tol, "Zero Division Error", 0,0,0
        train_pred = model.predict(X_train_all)
        test_pred = model.predict(X_test)

        train_pred = np.where(train_pred == -1, 0, train_pred)
        test_pred = np.where(test_pred == -1, 0, test_pred)
        y_train_all = np.where(y_train_all == -1, 0, y_train_all)

        train_acc.append(accuracy_score(train_pred, y_train_all))
        test_acc.append(accuracy_score(test_pred, y_test))

        train_mse.append(mse(train_pred, y_train_all))
        test_mse.append(mse(test_pred, y_test))

    plt.rcParams["figure.figsize"] = (5, 5)

    fig, axs = plt.subplots(1, 1)

    samples = (np.arange(len(train_mse))+1) * int(len(X_train_all)/epochs)
    # axs[0].plot(samples, train_acc, label="train")
    # axs[0].plot(samples, test_acc, label="test")
    # axs[0].legend()
    # axs[0].set_xlabel("samples")
    # axs[0].set_ylabel("accuracy")

    axs.plot(samples, train_mse, label="train")
    axs.plot(samples, test_mse, label="test")
    axs.legend()

    axs.set_xlabel("samples")
    axs.set_ylabel("mse")
    plt.title("{} kernel, C={}, degree={}".format(kernel, C, degree))
    plt.savefig("results_monks/3/{}monk{}_c{}_p{}_{}.png".format(index,monk_n,C,degree,kernel))

    return monk_n, kernel, gamma, degree, C, tol, train_mse, test_mse, train_acc, test_acc

best_configs1 = [(1600,5),(3200,9),(800,6),(800,7),(400,9)]
best_configs2 = [(12800,10),(12800,11),(6400,13),(25600,9),(6400,13)]
best_configs3 = [('rbf',100,1),('poly',10,7),('poly',50,3),('poly',5,12),('poly',100,3)]
# experiment(1, "poly", 'scale', 3, 100, 1e-3)
for i, (k,c,p) in enumerate(best_configs3):
    experiment(3, k,'auto',p,c,1e-7, index=i)
# experiment(3, 'poly','scale',5,10,1e-1)