from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np

from SVR import SVR
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from sklearn import svm, preprocessing

from utils import plot_error, plot_sv_number

np.random.seed(42)
n_features = 300
X, y = make_regression(n_samples=5000, n_features=n_features)

X = preprocessing.StandardScaler().fit(X).transform(X)
y = 2 * (y - min(y)) / (max(y) - min(y)) - 1

print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

C = 1
kernel = 'rbf'
eps = 0.1
gamma = 'scale'

model = SVR(C=C, kernel=kernel, eps=eps, gamma=gamma)
train_err = []
test_err = []

sv = []
sv_sota = []

model_sota = svm.SVR(C=C, kernel=kernel, epsilon=eps, gamma=gamma)
train_err_sota = []
test_err_sota = []

batch_size = int(len(X_train) / 10)

print(batch_size)

for i in range(0, int(len(X) / batch_size)):
    print("i={}----------------------------------------------------------------------".format(i))
    bs = (i + 1) * batch_size

    n_sv, alphas, indices = model.train(X_train[:bs], y_train[:bs])
    prediction = model.predict(X_train)

    train_err.append(mse(prediction, y_train))
    test_err.append(mse(model.predict(X_test), y_test))

    model_sota.fit(X_train[:bs], y_train[:bs])
    prediction = model_sota.predict(X_train)
    # support_vector_indices = np.where(np.abs(model_sota.decision_function(X_train[:bs])) <= 1 + 1e-6)[0]
    # support_vectors = (X_train[:bs])[support_vector_indices]
    # n_sv_sota = len(support_vectors)

    train_err_sota.append(mse(prediction, y_train))
    test_err_sota.append(mse(model_sota.predict(X_test), y_test))
    # input()
    # sv.append(n_sv)
    # sv_sota.append(n_sv_sota)
    # print("data: {}, support_vectors: {}, smallest: {}, greatest: {},".format(bs, n_sv, min(alphas), max(alphas), ))
    # print("data: {}, support_vectors: {}, smallest: {}, greatest: {}, ".format(bs, n_sv_sota, min(alphas), max(alphas), ))
    # input()
plot_error(train_err, test_err, "mySVR {} C={} eps={}".format(kernel, C, eps))
# print(test_err)
plot_error(train_err_sota, test_err_sota, "sklearn {} C={} eps={}".format(kernel, C, eps))
# plot_sv_number(sv)
# plot_sv_number(sv_sota)
