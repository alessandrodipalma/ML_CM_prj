from numpy import sqrt, square, sum, mean, subtract
import numpy as np

def mean_euclidean_error(y_pred, y_d):
    return mean(sqrt(sum(square(subtract(y_pred, y_d)), axis=1)))

def mean_squared_error(y_pred, y_d):
    return mean(sum(square(subtract(y_pred, y_d)), axis=1))

def min_max_scale(min_max, data, max=None, min=None, standardize=True):
    """
    Method to scale in a specific range of values trought minmax scaling
    :param min_max: tuple (min, max)  describing the final min e max of the scaled data
    :param data: np.array to scale
    :param min: Optional. Set a minimum value instead of getting it from data
    :param max: Optional. Set a minimum value instead of getting it from data
    :return:
    """
    if min is None:
        min = np.min(data)
    if max is None:
        max = np.max(data)

    if standardize:
        data_std = ((data - min) / (max - min))
    else:
        data_std = data
    data_scaled = data_std * (min_max[1]-min_max[0]) + min_max[0]
    return data_scaled, (min, max)

