import numpy as np
from nltk import word_tokenize
from sklearn.model_selection import train_test_split

from tfidf import TfidfModel
from utils.loading import load_glove_embeddings, load_data
from utils.plotting import plot


class SumEmbeddingsModel(TfidfModel):
    def __init__(self, samples, labels, word_embeddings):
        self.word_embeddings = word_embeddings
        self.transform(samples, labels)
        self.train()
        plot(self.x_test, self.y_test)

    def transform(self, samples, labels):
        x = np.array(
            [np.sum(np.array([self.word_embeddings[j] for j in word_tokenize(i)], dtype=np.float), axis=0) for i in
             samples])
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, labels, random_state=1,
                                                                                test_size=0.2)


if __name__ == '__main__':
    samples, labels = load_data()
    t = SumEmbeddingsModel(samples, labels, load_glove_embeddings())
