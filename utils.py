from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


def plot_error(train, test, title=""):
    plt.plot(range(0, len(test)), test, c='r', label='test')
    plt.plot(range(0, len(train)), train, c='b', label='train')
    plt.title(title)
    plt.yscale('linear')
    plt.legend(loc='upper right')
    # plt.yscale('log')
    plt.show()

def plot_sv_number(sv):

    plt.plot(range(0, len(sv)), sv, c='b', label='train')
    plt.title('number of support vectors')
    plt.show()


def generate_regression_from_feature_sample_dict(feature_samples_dict, n_problems):
    all_problems=[]
    for i, d in enumerate(feature_samples_dict):
        problems = []
        for j in range(n_problems):
            X, y = make_regression(n_samples=d['samples'], n_features=d['features'])
            X = preprocessing.StandardScaler().fit(X).transform(X)
            y = 2 * (y - min(y)) / (max(y) - min(y)) - 1
            problems.append(train_test_split(X, y, test_size=0.33))
        all_problems.append(problems)
    return all_problems