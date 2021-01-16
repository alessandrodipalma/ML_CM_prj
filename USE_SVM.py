from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

from svm import SVM
from sklearn.metrics import mean_squared_error as mse

from sklearn.svm import SVC


np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=4,
                           n_classes=2, n_clusters_per_class=1)
# y = y / 10 + np.full(len(y), 0.1)
y = np.where(y == 0, 0, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

C=0.0001

def plot_error(train, test, title=""):

    plt.plot(range(0, len(test)), test, c='r', label='test')
    plt.plot(range(0, len(train)), train, c='b', label='train')
    plt.title("C={}".format(C))
    plt.legend(loc='upper right')
    plt.show()

def plot_sv_number(sv):

    plt.plot(range(0, len(sv)), sv, c='b', label='train')
    plt.title = 'number of support vectors'
    plt.show()

model = SVM(C=C)
train_err = []
test_err = []

sv = []

model_sota = SVC(C=C)
train_err_sota = []
test_err_sota = []

batch_size = 50
for i in range(0, int(len(X)/batch_size)):
    print("i={}----------------------------------------------------------------------".format(i))
    bs = (i+1)*batch_size
    n_sv = model.train(X_train[:bs], y_train[:bs])
    train_err.append(mse(model.predict(X_train), y_train))
    test_err.append(mse(model.predict(X_test), y_test))

    model_sota.fit(X_train[:bs], y_train[:bs])

    train_err_sota.append(mse(model_sota.predict(X_train), y_train))
    test_err_sota.append(mse(model_sota.predict(X_test), y_test))

    sv.append(n_sv)




plot_error(train_err, test_err)
# plot_error(train_err_sota, test_err_sota)
# plot_sv_number(sv)