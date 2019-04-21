from gensim.models import KeyedVectors
from nltk.tag import pos_tag
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

from statistics import mean
import sys
import time

from tweet_preprocessor import preprocess, stopwords

max_data_size = 250
max_cluster_size = 125

class TwitterKMeans:

    def __init__(self, model, init_clusters=3,
                 fading=0.85, active_thresh=0.25,
                 min_silhouette=0.1, split_factor=0.575,
                 merge_factor=1.125):
        self.__model = model

        self.__fading = fading
        self.__active_thresh = active_thresh
        self.__min_silhouette = min_silhouette
        self.__split_factor = split_factor
        self.__merge_factor = merge_factor

        self.__has_splitted = False
        self.__has_merged = False

        self.__tweets = None

        self.__clusterer = KMeans(n_clusters=init_clusters)
        self.__mini_clusterer = KMeans(n_clusters=2)
        self.__centroids = None

        self.__pca = None
        self.__scaler = StandardScaler()

        self.__davies_bouldin_score = None
        self.__silhouette_score = None
        self.__silhouette_scores = None
        self.__distance_matrix = None
    
    """""""""""""""""""""""""""
    Main clustering procedures
    """""""""""""""""""""""""""
    def cluster(self, tweets):
        start_time = time.time()

        self.__init = False

        tweets = preprocess(tweets)

        if self.__tweets is None or self.__centroids is None:
            self.__init_clusters(tweets)
        else:
            if self.__has_splitted or self.__has_merged:
                self.__has_splitted = self.__has_merged = False
            else:
                self.__has_splitted = self.__try_split()
                if not self.__has_splitted:
                    self.__has_merged = self.__try_merge()
            self.__increment_clusters(tweets)

        """""""""
        Evaluation
        """""""""
        self.__evalute_clusters()

    def __init_clusters(self, new_tweets, init=True):
        self.__tweets = new_tweets
        self.__tweets['ttl'] = 1

        tweet_vectors = [self.__create_vector(tweet) for tweet in self.__tweets['cleanText'].values]
        
        clustering = self.__clusterer.fit(tweet_vectors) 
        self.__tweets['label'] = clustering.labels_

        new_centroids = []
        for label in range(len(clustering.cluster_centers_)):
            tweets = self.__tweets[(self.__tweets['label'] == label)
                                   & (self.__tweets['ttl'] > self.__active_thresh)]
            if len(tweets.index) > 0:
                vectors = [self.__create_vector(tweet) for tweet in tweets['cleanText'].values]
                centroid = np.mean(vectors, axis=0).tolist()

                new_centroids.append(centroid)

        self.__centroids = new_centroids
    
    def __increment_clusters(self, new_tweets):
        new_tweets['ttl'] = 1

        tweet_vectors = [self.__create_vector(tweet) for tweet in new_tweets['cleanText'].values]

        tree = KDTree(self.__centroids)
        active_labels = []
        labels = []
        for vector in tweet_vectors:
            closest_label = self.__centroids.index(self.__centroids[tree.query(vector)[1]])
            if closest_label not in active_labels:
                active_labels.append(closest_label)
            labels.append(closest_label)
        new_tweets['label'] = labels

        is_active = self.__tweets['label'].isin(active_labels)
        self.__tweets.loc[is_active, 'ttl'] = 1
        self.__tweets.loc[~is_active, 'ttl'] = self.__tweets['ttl'] * self.__fading

        self.__tweets = self.__tweets.append(new_tweets)

        new_centroids = []
        for label in range(len(self.__centroids)):
            tweets = self.__tweets[(self.__tweets['label'] == label)
                                   & (self.__tweets['ttl'] > self.__active_thresh)]
            if len(tweets.index) > 0:
                vectors = [self.__create_vector(tweet) for tweet in tweets['cleanText'].values]
                centroid = np.mean(vectors, axis=0).tolist()

                new_centroids.append(centroid)

        self.__centroids = new_centroids
    
    def __try_split(self):
        labels_to_split = [label for label, score in enumerate(self.__silhouette_scores)
                           if score > self.__min_silhouette
                           and score < self.__silhouette_score * self.__split_factor]
        if len(labels_to_split) == 0:
            return False
        
        for label in labels_to_split:
            tweets = self.__tweets[(self.__tweets['label'] == label)
                                   & (self.__tweets['ttl'] > self.__active_thresh)]
            tweet_vectors = [self.__create_vector(tweet) for tweet in tweets['cleanText'].values]

            clustering = self.__mini_clusterer.fit(tweet_vectors)
            tweets['label'] = clustering.labels_
            tweets['label'] = [label if lbl == 0 else label + len(self.__centroids) - 1
                               for lbl in tweets['label'].values]
            self.__tweets.update(tweets)

            self.__centroids[label] = clustering.cluster_centers_[0].tolist()
            for center in clustering.cluster_centers_[1:]:
                self.__centroids.append(center.tolist())
            
            if len(self.__centroids) >= 8:
                break
        
        return True
    
    def __try_merge(self):
        if len(self.__centroids) <= 2 or self.__davies_bouldin_score < self.__merge_factor:
            return False
        
        closest_clusters = np.unravel_index(np.argmin(self.__distance_matrix, axis=None),
                                            self.__distance_matrix.shape)
        label_1, label_2 = (closest_clusters[0], closest_clusters[1]) if closest_clusters[0] < closest_clusters[1] \
                           else (closest_clusters[1], closest_clusters[0])

        def __update(label):
            if label == label_1 or label == label_2:
                return label_1
            elif label > label_2:
                return label - 1
            else:
                return label

        self.__tweets['label'] = self.__tweets['label'].apply(__update)

        tweets = self.__tweets[(self.__tweets['label'] == label_1)
                               & (self.__tweets['ttl'] > self.__active_thresh)]
        tweet_vectors = [self.__create_vector(word) for word in tweets['cleanText'].values]

        self.__centroids[label_1] = np.mean(tweet_vectors, axis=0).tolist()
        self.__centroids = self.__centroids[:label_2] + self.__centroids[label_2 + 1:]
    
    """""""""""""""
    Helper methods
    """""""""""""""
    
    def __create_vector(self, tweet):
        def __w2v(word):
            try:
                return self.__model[word]
            except KeyError:
                return [0.0] * self.__model.vector_size

        word_vectors = [__w2v(word) for word in tweet.split()]

        return np.mean(word_vectors, axis=0).tolist()
    
    def eval_results(self):
        return self.__davies_bouldin_score, self.__silhouette_score, self.__silhouette_scores
    
    def __evalute_clusters(self):
        active = self.__tweets['ttl'] > self.__active_thresh
        X = [self.__create_vector(tweet) for tweet in self.__tweets[active]['cleanText'].values]
        labels = self.__tweets[active]['label'].values

        self.__davies_bouldin_score = davies_bouldin_score(X, labels)
        self.__silhouette_score = silhouette_score(X, labels)
        silhouette_scores = silhouette_samples(X, labels)
        self.__silhouette_scores = []
        for label in range(len(self.__centroids)):
            score = np.mean([score for idx, score in enumerate(silhouette_scores)
                             if label == labels[idx]])
            self.__silhouette_scores.append(score)
        self.__silhouette_score = mean(self.__silhouette_scores)
        
        self.__distance_matrix = squareform(pdist(self.__centroids))
        self.__distance_matrix[self.__distance_matrix == 0.0] = sys.maxsize
