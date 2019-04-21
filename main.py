from utils.loading import load_glove_embeddings, load_data
from avg_embedding import AvgEmbeddingsModel
from sum_embedding import SumEmbeddingsModel
from tfidf import TfidfModel

samples, labels = load_data()
glove = load_glove_embeddings()

avg_embeddings = AvgEmbeddingsModel(samples, labels, glove)
sum_embeddings = SumEmbeddingsModel(samples, labels, glove)
tfidf_model = TfidfModel(samples, labels)