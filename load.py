import glob
from random import shuffle

import matplotlib.pyplot as plt
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
    clf.fit(samples[:1900], labels[:1900])
    print(clf.score(samples[1900:], labels[1900:]))
    for i in range(1900, 2000):
        plt.plot(i, labels[i], 'g+') if labels[i] == 1 else plt.plot(i, labels[i], 'rx')
    plt.plot(range(1900, 2000), [clf.predict_proba(i)[0][1] for i in samples[1900:]], 'blue')
    plt.show()
    return clf


def predict_from_file(vectorizer, clf):
    file = open("testing")
    testFile = file.read()
    X = vectorizer.transform([testFile])
    return clf.predict(X)


if __name__ == "__main__":
    samples, labels = ViewFiles()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(samples)
    clf = model(X, labels)
    # print(predict_from_file(vectorizer, clf))

