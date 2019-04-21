import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import TruncatedSVD


def plot(samples, labels):
    fig = plt.figure()
    ax = Axes3D(fig)
    data3d = TruncatedSVD(n_components=3).fit_transform(samples)
    # print(data3d.shape)
    for i in range(len(labels)):
        if labels[i] == 1:
            ax.scatter(data3d[:, 0][i], data3d[:, 1][i], data3d[:, 2][i], c='green')
        else:
            ax.scatter(data3d[:, 0][i], data3d[:, 1][i], data3d[:, 2][i], c='red')
    plt.show()