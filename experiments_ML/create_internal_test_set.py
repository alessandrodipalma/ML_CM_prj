import random

import pandas as pd

df = pd.read_csv('cup 2021/ML-CUP21-TR.csv', header=None)

indices = sorted(random.sample(range(len(df)), int(len(df)*0.2)))
print(df.loc[df.index[indices]].to_csv('cup 2021/ML-CUP21-INT-TS.csv', header=None, index=False))
print(df.loc[set(df.index) - set(indices)].to_csv('cup 2021/ML-CUP21-INT-TR.csv', header=None, index=False))