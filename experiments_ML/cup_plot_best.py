import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from tabulate import tabulate

from Cplex_Solver import CplexSolver
from SVR import SVR

from experiments_ML.metrics import mean_squared_error, mean_euclidean_error
from experiments_ML.Scaler import Scaler
from experiments_ML.load_cup_ds import load_cup_int_train, load_cup_int_test

X, y = load_cup_int_train()
X_int_test, y_int_test = load_cup_int_test()
print(X.shape, y.shape)

kf = KFold(n_splits=5)
def experiment(C, alpha_tol, eps, kernel, gamma, degree, tol):
    train_mse = []
    test_mse = []
    train_mee = []
    test_mee = []

    for train_ind, test_ind in kf.split(X):

        X_train = X[train_ind, :]
        y_train = y[train_ind, :]
        X_test = X[test_ind, :]
        y_test = y[test_ind, :]
        scaler = Scaler()
        # print(train_ind, test_ind, X_train.shape, y_test.shape)
        X_train, scaled_y_train, X_test, scaled_y_test = scaler.scale(X_train, y_train, X_test, y_test)

        # solver = GVPM(ls=GVPM.LineSearches.BACKTRACK, n_min=2, tol=tol, lam_low=1e-2, a_max=1e10,
        #               a_min=1, plots=False, verbose=False, proj_tol=1e-7, max_iter=1e5)
        solver = CplexSolver(tol = tol)
        model = SVR(solver=solver, C=C, kernel=kernel, eps=eps, gamma=gamma, degree=degree, alpha_tol=alpha_tol)
        try:
            print("fitting...")
            model.fit(X_train, scaled_y_train)
            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)
            # pred_int_test = model.predict(X_int_test)

            pred_train = scaler.scale_back(pred_train)
            pred_test = scaler.scale_back(pred_test)

            train_mse.append(mean_squared_error(y_train, pred_train))
            test_mse.append(mean_squared_error(y_test, pred_test))
            train_mee.append(mean_euclidean_error(y_train, pred_train))
            test_mee.append(mean_euclidean_error(y_test, pred_test))
            print(test_mee, train_mee)
        except Exception:
            print(kernel, C, alpha_tol, eps, gamma, degree, tol, Exception.__name__)

            return kernel, C, alpha_tol, eps, gamma, degree, tol, 0, 0, 0, Exception.__name__

    # print(train_err, test_err, train_mae, test_mae)
    train_mse = remove_nans(train_mse)
    test_mse = remove_nans(test_mse)
    train_mee = remove_nans(train_mee)
    test_mee = remove_nans(test_mee)

    r = kernel, C, alpha_tol, eps, gamma, degree, tol, \
        np.mean(train_mse, ), np.std(train_mse), np.mean(test_mse), np.std(test_mse), \
        np.mean(train_mee), np.std(train_mee), np.mean(test_mee), np.std(test_mee)
    with open("cup 2021/temp_check_bests.csv", "a", encoding="utf-8") as out_file:
        out_file.write(tabulate([r], tablefmt='tsv'))
        out_file.write("\n")
    print(kernel, C, alpha_tol, eps, gamma, degree, tol, np.mean(train_mee), np.mean(test_mee))

    return r


def remove_nans(train_mse):
    train_mse = np.array(train_mse)
    train_mse = train_mse[np.logical_not(np.isnan(train_mse))]
    return train_mse


def experiment_nokf(C, alpha_tol, eps, kernel, gamma, degree, tol, index="", epochs=1, plot=False):
    train_mse = []
    test_mse = []
    train_mee = []
    test_mee = []

    X_train = X
    y_train = y
    X_test = X_int_test
    y_test = y_int_test
    scaler = Scaler()
    # print(train_ind, test_ind, X_train.shape, y_test.shape)
    X_train, scaled_y_train, X_test, scaled_y_test = scaler.scale(X_train, y_train, X_test, y_test)

    # solver = GVPM(ls=GVPM.LineSearches.BACKTRACK, n_min=2, tol=tol, lam_low=1e-2, a_max=1e10,
    #               a_min=1, plots=False, verbose=False, proj_tol=1e-7, max_iter=1e5)
    solver = CplexSolver(tol = tol)
    model = SVR(solver=solver, C=C, kernel=kernel, eps=eps, gamma=gamma, degree=degree, alpha_tol=alpha_tol)

    epochs = epochs
    for k in range(epochs):
        print("fitting...")
        bs = int(len(X_train) / epochs) * (k + 1)
        print("samples", bs)
        X_train_batch = X_train[:bs]
        scaled_y_train_batch = scaled_y_train[:bs]
        print(X_train_batch.shape, scaled_y_train_batch.shape)

        model.fit(X_train_batch, scaled_y_train_batch)
        # model.save("cup 2021/final_model.bin")
        # model = SVR(solver=solver, C=C, kernel=kernel, eps=eps, gamma=gamma, degree=degree, alpha_tol=alpha_tol)
        # model.set_params("cup 2021/final_model.bin")
        pred_train = model.predict(X_train_batch)
        pred_test = model.predict(X_test)

        pred_train = scaler.scale_back(pred_train)
        pred_test = scaler.scale_back(pred_test)

        train_mse.append(mean_squared_error(y_train[:bs], pred_train))
        test_mse.append(mean_squared_error(y_test, pred_test))
        train_mee.append(mean_euclidean_error(y_train[:bs], pred_train))
        test_mee.append(mean_euclidean_error(y_test, pred_test))
        print(test_mee, train_mee)

    r = kernel, C, alpha_tol, eps, gamma, degree, tol, \
        np.mean(train_mse, ), np.std(train_mse), np.mean(test_mse), np.std(test_mse), \
        np.mean(train_mee), np.std(train_mee), np.mean(test_mee), np.std(test_mee)
    with open("cup 2021/best_results_int_test.csv", "a", encoding="utf-8") as out_file:
        out_file.write(tabulate([r], tablefmt='tsv'))
        out_file.write("\n")

    if plot:
        plt.rcParams["figure.figsize"] = (5, 5)
        fig, axs = plt.subplots(1, 1)
        samples = (np.arange(len(train_mse))+1) * int(len(X_train)/epochs)
        axs.plot(samples, train_mse, label="train")
        axs.plot(samples, test_mse, label="test")
        axs.legend()
        axs.set_xlabel("samples")
        axs.set_ylabel("mse")
        plt.title("{} kernel, C={}, degree={}".format(kernel, C, degree))
        plt.savefig("cup 2021/learning_curves/c{}_p{}_{}.png".format(C,degree,kernel))

    return r

best_configs = [
    # ('poly',2560,1e-1,1e-2,'scale',5,1e-3),
    # ('poly',80,1e-7,1e-3,'scale',3,1e-3),
    # ('poly',2560,1e-1,1e-2,'scale',3,1e-3),
    # ('rbf',2560,1e-3,1e-3,'auto',1,1e-5),
    ('poly',2560, 1e-1, 1e-2,'auto',3,1e-3),
    # ('rbf',1280,1e-3,1e-3,'auto',1,1e-5),
]

# for i, (kernel, C, alpha_tol, eps, gamma, degree, tol) in enumerate(best_configs):
#     experiment_nokf(C, alpha_tol, eps, kernel, gamma, degree, tol, epochs=1)

table = Parallel(n_jobs=4, verbose=11)(delayed(experiment_nokf)(C, alpha_tol, eps, kernel, gamma, degree, tol)
                                                                for (kernel, C, alpha_tol, eps, gamma, degree, tol) in best_configs)