from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

from svm import SVM
from sklearn.metrics import mean_squared_error as mse




np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=8,
                           n_classes=3, n_clusters_per_class=1, shift=0.1, scale=0.1)
y = y / 10 + np.full(len(y), 0.1)
# y = np.where(y == 0, 0.1, y).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

C=0.005
model = SVM(C=C)
train_err = []
test_err = []
for i in range(25, len(X), 50):
    model.train(X_train[:i], y_train[:i])
    train_err.append(mse([model.predict(X_train[i]) for i in range(len(X_train))], y_train))
    test_err.append(mse([model.predict(X_test[i]) for i in range(len(X_test))], y_test))


def plot_error(train, test):
    plt.title = "C={}".format(C)
    plt.plot(range(0, len(test)), test, c='r', label='test')
    plt.plot(range(0, len(train)), train, c='b', label='train')

    plt.legend(loc='upper right')
    plt.show()

plot_error(train_err, test_err)