import numpy as np

from GVPM import GVPM
from experiments_ML.load_monk import load_monk
from SVM import SVM
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from utils import plot_error

np.random.seed(42)
# n_features = 10
# X, y = make_classification(n_samples=400, n_features=n_features, n_classes=2, n_clusters_per_class=1, n_redundant=0)
X_train, y_train = load_monk(3, 'train')
X_test, y_test = load_monk(3, 'test')
# y = y / 10 + np.full(len(y), 0.1)

y_train = np.where(y_train == 0, -1, y_train)
y_test = np.where(y_test == 0, -1, y_test)
# print(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

C = 100
kernel = 'poly'

solver = GVPM(ls=GVPM.LineSearches.BACKTRACK, n_min=2, tol=1e-3, lam_low=1e-3, plots=False, proj_tol=1e-3)
model = SVM(solver = solver,
            # exact_solver=CplexSolver(tol=tol, verbose=False),
            C=C, kernel=kernel, gamma="auto", degree=9)
train_err = []
test_err = []

sv = []
sv_sota = []

model_sota = SVC(C=C, kernel=kernel)
train_err_sota = []
test_err_sota = []

batch_size = int(len(X_train) / 10)

for i in range(0, int(len(X_train) / batch_size)):
    print("i={}----------------------------------------------------------------------".format(i))
    bs = (i + 1) * batch_size

    n_sv, alphas, indices = model.train(X_train[:bs], y_train[:bs])
    prediction = model.predict(X_train)

    train_err.append(accuracy_score(prediction, y_train))
    test_err.append(accuracy_score(model.predict(X_test), y_test))

    model_sota.fit(X_train[:bs], y_train[:bs])
    prediction = model_sota.predict(X_train)
    support_vector_indices = np.where(np.abs(model_sota.decision_function(X_train[:bs])) <= 1 + 1e-6)[0]
    support_vectors = (X_train[:bs])[support_vector_indices]
    n_sv_sota = len(support_vectors)

    train_err_sota.append(accuracy_score(prediction, y_train))
    test_err_sota.append(accuracy_score(model_sota.predict(X_test), y_test))

    sv.append(n_sv)
    sv_sota.append(n_sv_sota)
    print("data: {}, support_vectors: {}, smallest: {}, greatest: {},".format(bs, n_sv, min(alphas), max(alphas), ))
    print(
        "data: {}, support_vectors: {}, smallest: {}, greatest: {}, ".format(bs, n_sv_sota, min(alphas), max(alphas), ))

plot_error(train_err, test_err, "mySVM " + kernel)
plot_error(train_err_sota, test_err_sota, "sklearn " + kernel)
# plot_sv_number(sv)
# plot_sv_number(sv_sota)
