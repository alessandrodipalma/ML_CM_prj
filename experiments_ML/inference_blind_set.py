import numpy as np

from GVPM import GVPM
from Cplex_Solver import CplexSolver
from SVR import SVR
from experiments_ML.load_cup_ds import load_cup_test, load_cup_int_train
from experiments_ML.Scaler import Scaler
import pandas as pd

blind_test = load_cup_test()

X_train, y_train = load_cup_int_train()

scaler = Scaler()
X_train, scaled_y_train = scaler.fit_and_scale(X_train, y_train)
X_test = scaler.scale_X(blind_test)

kernel, C, alpha_tol, eps, gamma, degree, tol = ('poly',2560,1e-1,1e-2,'scale',5,1e-3)
solver = CplexSolver(tol = tol)
model = SVR(solver=solver, kernel= kernel, C=C, alpha_tol=alpha_tol, eps=eps, gamma=gamma, degree=degree)
model.set_params("cup 2021/final_model_full_train.bin")
# model.fit(X_train, scaled_y_train)
# model.save("cup 2021/final_model_full_train.bin")
predictions = model.predict(X_test)
predictions = scaler.scale_back(predictions)


predictions = pd.DataFrame(predictions)
predictions.index = predictions.index + 1
predictions.to_csv("cup 2021/blind_test_cup_predictions.csv")
