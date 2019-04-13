from sklearn.feature_extraction.text import TfidfVectorizer
import glob
import numpy as np

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
    return samples, labels
    # print("Negatives: ", list_of_Negfiles)
    # print("Positives: ", list_of_Posfiles)
def tf_idf(samples):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(samples)
    return X
def main():
    samples, labels = ViewFiles()
    # for i in samples:
    #     print(i)
    #     print()
    print(tf_idf(samples)[0])

if __name__ == "__main__":
    main()