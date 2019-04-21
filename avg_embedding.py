import numpy as np
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sum_embedding import SumEmbeddingsModel


class AvgEmbeddingsModel(SumEmbeddingsModel):
    def transform(self, samples, labels):
        x = np.array(
            [np.average(np.array([self.word_embeddings[j] for j in word_tokenize(i)], dtype=np.float), axis=0) for i in
             samples])
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, labels, random_state=1,
                                                                                test_size=0.2)
