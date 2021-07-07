from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import libsvm.svm as SVM_lib
from svm import SVM
from sklearn.metrics import mean_squared_error as mse
from sklearn.svm import SVC

from utils import plot_error, plot_sv_number

np.random.seed(42)
n_features = 10
X, y = make_classification(n_samples=400, n_features=n_features, n_classes=2, n_clusters_per_class=1, n_redundant=0)
# y = y / 10 + np.full(len(y), 0.1)
y = np.where(y == 0, -1, y)
# print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

C = 1
kernel = 'poly'

model = SVM(C=C, kernel=kernel)
train_err = []
test_err = []

sv = []
sv_sota = []

model_sota = SVC(C=C, kernel=kernel)
train_err_sota = []
test_err_sota = []

batch_size = int(len(X_train) / 10)

for i in range(0, int(len(X) / batch_size)):
    print("i={}----------------------------------------------------------------------".format(i))
    bs = (i + 1) * batch_size

    n_sv, alphas, indices = model.train(X_train[:bs], y_train[:bs], sigma=1 / (n_features * X_train.var()))
    prediction = model.predict(X_train)

    train_err.append(mse(prediction, y_train))
    test_err.append(mse(model.predict(X_test), y_test))

    model_sota.fit(X_train[:bs], y_train[:bs])
    prediction = model_sota.predict(X_train)
    support_vector_indices = np.where(np.abs(model_sota.decision_function(X_train[:bs])) <= 1 + 1e-6)[0]
    support_vectors = (X_train[:bs])[support_vector_indices]
    n_sv_sota = len(support_vectors)

    train_err_sota.append(mse(prediction, y_train))
    test_err_sota.append(mse(model_sota.predict(X_test), y_test))

    sv.append(n_sv)
    sv_sota.append(n_sv_sota)
    print("data: {}, support_vectors: {}, smallest: {}, greatest: {},".format(bs, n_sv, min(alphas), max(alphas), ))
    print(
        "data: {}, support_vectors: {}, smallest: {}, greatest: {}, ".format(bs, n_sv_sota, min(alphas), max(alphas), ))

plot_error(train_err, test_err, "mySVM " + kernel)
plot_error(train_err_sota, test_err_sota, "sklearn " + kernel)
# plot_sv_number(sv)
# plot_sv_number(sv_sota)
