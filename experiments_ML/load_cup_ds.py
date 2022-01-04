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


