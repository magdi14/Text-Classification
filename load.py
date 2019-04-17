import glob
from random import shuffle, seed

import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def ViewFiles():
    list_of_Negfiles = glob.glob('./corpus/neg/*.txt')
    list_of_Posfiles = glob.glob('./corpus/pos/*.txt')
    samples = []
    labels = []
    for i in list_of_Negfiles:
        f = open(i)
        samples.append(f.read())
        labels.append(0)
    for i in list_of_Posfiles:
        f = open(i)
        samples.append(f.read())
        labels.append(1)
    samples_labels = list(zip(samples, labels))
    shuffle(samples_labels)
    samples, labels = zip(*samples_labels)
    return samples, labels


def model(samples, labels):
    clf = LogisticRegression(solver='lbfgs')
    clf.fit(samples[:1800], labels[:1800])
    print("Accuracy:", clf.score(samples[1800:], labels[1800:]))
    data2d = TruncatedSVD(n_components=2).fit_transform(samples)
    print(data2d.shape)
    for i in range(len(labels[1800:])):
        if labels[i] == 1:
            plt.scatter(x=data2d[:, 0][i], y=data2d[:, 1][i], c='green')
        else:
            plt.scatter(x=data2d[:, 0][i], y=data2d[:, 1][i], c='red')
    plt.show()
    return clf


def predict_from_file(vectorizer, clf):
    file = open("testing")
    testFile = file.read()
    X = vectorizer.transform([testFile])
    return "Positive" if clf.predict(X) == [1] else "Negative"


if __name__ == "__main__":
    seed(2)
    samples, labels = ViewFiles()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(samples)
    clf = model(X, labels)
    # print(predict_from_file(vectorizer, clf))
