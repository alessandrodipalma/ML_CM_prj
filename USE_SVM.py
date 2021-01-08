from matplotlib import pyplot
from sklearn.datasets import make_classification
import numpy as np

from svm import SVM
from sklearn.metrics import mean_squared_error as mse

def plot_error(errors):
    pyplot.scatter(range(0, len(errors)), errors)
    pyplot.title("error")
    pyplot.show()

np.random.seed(42)
np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=4,
                           n_classes=2, n_clusters_per_class=2)

y = np.where(y == 0, -1, y)

model = SVM(C=0.01)
errors=[]
for i in range(2,len(X), 50):
    model.train(X[:i], y[:i])
    err = mse([model.predict(X[i]) for i in range(len(X))], y)
    # print(err)
    errors.append(err)

plot_error(errors)