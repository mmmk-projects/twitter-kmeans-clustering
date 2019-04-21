from gensim.models import KeyedVectors
import pandas as pd

from clusterer import TwitterKMeans

update_size = 75

documents = pd.read_csv('./data/scraped_tweets.csv', dtype=object) \
              .dropna(subset=['text']) \
              .reset_index()
documents['cleanText'] = documents['text']
documents.sort_values(by=['time'], inplace=True)

model = KeyedVectors.load('./model/gensim_w2v.kv')

pca = None

clustered_docs = {}
clustered_word_count = {}
