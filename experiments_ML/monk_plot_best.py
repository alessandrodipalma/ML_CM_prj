import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

from gvpm import GVPM
from load_monk import load_monk
from svm import SVM
from sklearn.metrics import mean_squared_error as mse, accuracy_score


def experiment(monk_n, kernel, gamma, degree, C, tol):
    X_train_all, y_train_all = load_monk(monk_n, 'train')
    X_test, y_test = load_monk(monk_n, 'test')

    scaler = StandardScaler()
    X_train_all = scaler.fit_transform(X_train_all)
    X_test = scaler.transform(X_test)
    train_acc = []
    test_acc = []
    train_mse = []
    test_mse = []

    epochs = 15
    for k in range(epochs):

        bs = int(len(X_train_all)/epochs) * (k+1)
        print("samples", bs)
        X_train = X_train_all[:bs]
        y_train = y_train_all[:bs]
        print(X_train.shape, y_train.shape)
        y_train = np.where(y_train == 0, -1, y_train)


        solver = GVPM(ls=GVPM.LineSearches.BACKTRACK, n_min=2, tol=tol, lam_low=1e-3, proj_tol=1e-3)
        model = SVM(solver=solver,
                    # exact_solver=CplexSolver(tol=tol, verbose=False),
                    C=C, kernel=kernel, gamma=gamma, degree=degree)

        try:
            n_sv, alphas, indices = model.train(X_train, y_train)
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

    plt.rcParams["figure.figsize"] = (9, 5)
    fig, axs = plt.subplots(1, 2)
    samples = (np.arange(len(train_mse))+1) * int(len(X_train_all)/epochs)
    axs[0].plot(samples, train_acc, label="train")
    axs[0].plot(samples, test_acc, label="test")
    axs[0].legend()
    axs[0].set_xlabel("samples")
    axs[0].set_ylabel("accuracy")

    axs[1].plot(samples, train_mse, label="train")
    axs[1].plot(samples, test_mse, label="test")
    axs[1].legend()
    axs[1].set_xlabel("samples")
    axs[1].set_ylabel("mse")
    plt.savefig("plots/best_monk/scaled_monk_{}_learning_curve.png".format(monk_n))

    return monk_n, kernel, gamma, degree, C, tol, train_mse, test_mse, train_acc, test_acc

# experiment(1, "poly", 'scale', 3, 100, 1e-3)
experiment(2, 'rbf','auto',1,50,1e-1)
# experiment(3, 'poly','scale',5,10,1e-1)