import pandas as pd

def load_cup_train():
    df = pd.read_csv('monk/ML-CUP20-TR.csv')
    x = df.iloc[:, 1:].to_numpy()
    y = df.iloc[:, :2].to_numpy()
    return(x, y)


def load_cup_test():
    df = pd.read_csv('monk/ML-CUP20-TS.csv')
    x = df.iloc[:, 1:].to_numpy()
    return(x)