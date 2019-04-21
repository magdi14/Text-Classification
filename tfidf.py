from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils.loading import load_data
from utils.plotting import plot


class TfidfModel:
    def __init__(self, samples, labels):
        self.transform(samples, labels)
        self.train()
        plot(self.x_test, self.y_test)

    def transform(self, samples, labels):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.vectorizer.fit_transform(samples),
                                                                                labels, random_state=1, test_size=0.2)

    def train(self):
        self.clf = LogisticRegression(solver='lbfgs')
        self.clf.fit(self.x_train, self.y_train)
        print("Accuracy:", self.clf.score(self.x_test, self.y_test))

    def predict(self, text):
        X = self.vectorizer.transform([text])
        return "Negative" if self.clf.predict(X)[0] == 0 else "Positive"


if __name__ == "__main__":
    samples, labels = load_data()
    t = TfidfModel(samples, labels)
