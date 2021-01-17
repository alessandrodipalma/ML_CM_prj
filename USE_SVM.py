from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

from svm import SVM
from sklearn.metrics import mean_squared_error as mse

from sklearn.svm import SVC


np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=10,
                           n_classes=2, n_clusters_per_class=1)
# y = y / 10 + np.full(len(y), 0.1)
y = y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

C=0.005

def plot_error(train, test, title=""):

    plt.plot(range(0, len(test)), test, c='r', label='test')
    plt.plot(range(0, len(train)), train, c='b', label='train')
    plt.title("C={}, model={}".format(C, title))
    plt.legend(loc='upper right')
    plt.show()

def plot_sv_number(sv):

    plt.plot(range(0, len(sv)), sv, c='b', label='train')
    plt.title('number of support vectors')
    plt.show()

model = SVM(C=C)
train_err = []
test_err = []

sv = []

model_sota = SVC(C=C)
train_err_sota = []
test_err_sota = []

batch_size = int(len(X) / 10)
for i in range(0, int(len(X)/batch_size)):
    print("i={}----------------------------------------------------------------------".format(i))
    bs = (i+1)*batch_size
    n_sv, alphas = model.train(X_train[:bs], y_train[:bs])
    train_err.append(mse(model.predict(X_train), y_train))
    test_err.append(mse(model.predict(X_test), y_test))

    model_sota.fit(X_train[:bs], y_train[:bs])

    train_err_sota.append(mse(model_sota.predict(X_train), y_train))
    test_err_sota.append(mse(model_sota.predict(X_test), y_test))

    sv.append(n_sv)
    print("data: {}, support_vectors: {}, smallest: {}, greatest: {}".format(bs, n_sv, min(alphas), max(alphas)))




plot_error(train_err, test_err, "mySVM")
plot_error(train_err_sota, test_err_sota, "sklearn")
plot_sv_number(sv)