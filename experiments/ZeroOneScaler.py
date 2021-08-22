import numpy as np


class Scaler:
    def __init__(self):
        pass

    def bring_in_zeroone(self, a):
        self.miny = np.zeros(a.shape[1])
        self.maxy = np.zeros(a.shape[1])
        for c in range(a.shape[1]):
            self.miny[c] = min(a[:, c])
            self.maxy[c] = max(a[:, c])
            a[:, c] = 2 * (a[:, c] - np.full(a.shape[0], self.miny[c]))\
                      / (self.maxy[c] - self.miny[c]) - np.ones(a.shape[0])
        return a

    def revert_scaling(self, a):
        for c in range(a.shape[1]):
            a[:, c] = (a[:, c] + np.ones(a.shape[0])) * ((self.maxy[c] - self.miny[c]) / 2) + np.full(a.shape[0],self.miny[c])
        return a