import  numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from utils import plot_error
from SVM import SVM
from sklearn.metrics import mean_squared_error as mse
from sklearn.svm import SVC

np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=50,
                           n_classes=2, n_clusters_per_class=1)
y = np.where(y == 0, -1, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)




batch_size = int(len(X) / 10)

for k in SVM.KERNELS:
    print("KERNEL: {}".format(k))
    C = 1
    model = SVM(C=C, kernel=k)

    train_errs = []
    test_errs = []

    for i in range(0, int(len(X) / batch_size)):
        print("i={}----------------------------------------------------------------------".format(i))
        bs = (i + 1) * batch_size

        X_tr = X_train[:bs]
        y_tr = y_train[:bs]
        X_te = X_test[:bs]
        y_te = y_test[:bs]

        model.train(X_tr, y_tr)
        train_err = mse(model.predict(X_tr), y_tr)
        test_err = mse(model.predict(X_te), y_te)

        train_errs.append(train_err)
        test_errs.append(test_err)

    plot_error(train_errs, test_errs, title="{} kernel, C = {}".format(k, C))



