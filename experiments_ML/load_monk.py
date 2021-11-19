import pandas as pd


def load_monk(batch=1, subset="train"):
    df = pd.read_csv('monk/monks-{}.{}'.format(batch, subset), sep=' ', header=None)

    x = df.iloc[:, 2:-1].to_numpy()
    y = df.iloc[:, 1:2].to_numpy()
    y = y.reshape((y.shape[0]))
    print(x,y)
    return x, y

