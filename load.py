import glob
from random import shuffle

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
    # print("Negatives: ", list_of_Negfiles)
    # print("Positives: ", list_of_Posfiles)


def model(samples, labels):
    clf = LogisticRegression(solver='lbfgs')
    clf.fit(samples[:1900], labels[:1900])
    print(clf.score(samples[1900:], labels[1900:]))
    return clf


def predict_from_file(vectorizer, clf):
    pass


if __name__ == "__main__":
    samples, labels = ViewFiles()
    # print(labels)
    # for i in samples:
    #     print(i)
    #     print()
    # print(tf_idf(samples)[0])
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(samples)
    clf = model(X, labels)
