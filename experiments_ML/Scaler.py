from experiments_ML.metrics import min_max_scale


class Scaler:
    def __init__(self, scale_min=0, scale_max=1):
        self.scale_min = scale_min
        self.scale_max = scale_max

    def scale_back(self, y_pred):
        y_pred, (_, _) = min_max_scale((self.miny, self.maxy), y_pred, min=self.scale_min, max=self.scale_max, standardize=False)
        return y_pred


    def fit_and_scale(self, X_train, y_train):
        X_train, (self.minx, self.maxx) = min_max_scale((self.scale_min, self.scale_max), X_train)
        scaled_y_train, (self.miny, self.maxy) = min_max_scale((self.scale_min, self.scale_max), y_train)
        return X_train, scaled_y_train

    def scale_X(self, X):
        X, (_, _) = min_max_scale((self.scale_min, self.scale_max), X, min=self.minx, max=self.maxx)
        return X

    def scale_y(self, y):
        y, _ = min_max_scale((self.scale_min, self.scale_max), y, self.miny, self.maxy)
        return y

    def scale_only(self, X, y):
        return self.scale_X(X), self.scale_y(y)

    def scale(self, X_train, y_train, X_valid, y_valid):
        X_train, (self.minx, self.maxx) = min_max_scale((self.scale_min, self.scale_max), X_train)
        X_valid, (_, _) = min_max_scale((self.scale_min, self.scale_max), X_valid, min=self.minx, max=self.maxx)

        scaled_y_train, (self.miny, self.maxy) = min_max_scale((self.scale_min, self.scale_max), y_train)
        scaled_y_valid, _ = min_max_scale((self.scale_min, self.scale_max), y_valid, self.miny, self.maxy)

        return X_train, scaled_y_train, X_valid, scaled_y_valid