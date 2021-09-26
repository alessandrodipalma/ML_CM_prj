import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from SVR import SVR
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, euclidean_distances

from experiments_ML.ZeroOneScaler import Scaler
from GVPM import GVPM
from experiments_ML.load_cup_ds import load_cup_train

X_all, y_all = load_cup_train()
X_all = preprocessing.StandardScaler().fit(X_all).transform(X_all)
# print(X.shape, y.shape)


def experiment(config, C, eps, kernel, gamma, degree, tol):

    epochs = 15


    X, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.33)

    train_mse = []
    test_mse = []
    train_mae = []
    test_mae = []
    train_mee = []
    test_mee = []
    for k in range(epochs):
        bs = int(len(X_all) / epochs) * (k + 1)
        print("samples", bs)
        X_train = X[:bs]
        y_batch = y_train[:bs]
        y_scaler = Scaler()
        y_batch = y_scaler.bring_in_zeroone(y_batch)
        print(X_train.shape, X_test.shape, y_batch.shape, y_test.shape)

        solver = GVPM(ls=GVPM.LineSearches.BACKTRACK, n_min=2, tol=tol, lam_low=1e-3, plots=False, proj_tol=1e-3)
        model = SVR(solver=solver, C=C, kernel=kernel, eps=eps, gamma=gamma, degree=degree)

        model.train(X_train, y_batch)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        pred_train = y_scaler.revert_scaling(pred_train)
        pred_test = y_scaler.revert_scaling(pred_test)
        y_batch = y_scaler.revert_scaling(y_batch)

        train_mse.append(mse(y_batch, pred_train))
        test_mse.append(mse(y_test, pred_test))
        train_mae.append(mae(y_batch, pred_train))
        test_mae.append(mae(y_test, pred_test))
        train_mee.append(np.mean(euclidean_distances(y_batch, pred_train)))
        test_mee.append(np.mean(euclidean_distances(y_test, pred_test)))
        print(test_mee)

    plt.rcParams["figure.figsize"] = (13, 5)
    fig, axs = plt.subplots(1, 3)
    samples = (np.arange(len(train_mse))+1) * int(len(X_train)/epochs)
    axs[0].plot(samples, train_mse, label="train")
    axs[0].plot(samples, test_mse, label="test")
    axs[0].legend()
    axs[0].set_xlabel("samples")
    axs[0].set_ylabel("mse")

    axs[1].plot(samples, train_mee, label="train")
    axs[1].plot(samples, test_mee, label="test")
    axs[1].legend()
    axs[1].set_xlabel("samples")
    axs[1].set_ylabel("Mean Euclidean Distance")
    plt.savefig("plots/best_cup/cup_{}_learning_curve.png".format(config))

    axs[2].plot(samples, train_mae, label="train")
    axs[2].plot(samples, test_mae, label="test")
    axs[2].legend()
    axs[2].set_xlabel("samples")
    axs[2].set_ylabel("Mean Absolute Error")
    plt.savefig("plots/best_cup/cup_{}_learning_curve.png".format(config))

    #
    return kernel, C, eps, gamma, degree, tol, np.mean(train_mse), np.std(train_mse), np.mean(train_mse), np.std(
        test_mse), np.mean(train_mae), np.std(train_mae), np.mean(test_mae), np.std(test_mae), \
           np.mean(train_mee), np.std(test_mee), np.mean(test_mee), np.std(test_mee)

# experiment(1, 10, 1e-2, 'poly', 'scale', 7, 0.001)
# experiment(2, 1, 1e-3, 'linear', 'scale', 2, 0.1)
# experiment(3, 1, 1e-2, 'poly', 'scale', 3, 1e-3)
# experiment("4scaled", 0.01, 1e-4, 'poly', 'scale', 3, 1e-1)


