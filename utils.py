from matplotlib import pyplot as plt

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