from langdetect import detect
from nltk.corpus import stopwords, wordnet, words
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import pandas as pd
import preprocessor as p

import re
import warnings
warnings.filterwarnings("ignore")

words = set(words.words())
stopwords = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def remove_links_and_hashtags(tweet):
    return p.clean(str(tweet))

def to_lowercase(tweet):
    return tweet.lower()

def remove_contractions(tweet):
    tweet = re.sub(r'’', '\'', tweet)

    tweet = re.sub(r'won\'t', 'will not', tweet)
    tweet = re.sub(r'can\'t', 'can not', tweet)
    
    tweet = re.sub(r'\'s', ' is', tweet)
    tweet = re.sub(r'\'m', ' am', tweet)
    tweet = re.sub(r'\'re', ' are', tweet)
    tweet = re.sub(r'\'ve', ' have', tweet)
    tweet = re.sub(r'\'ll', ' will', tweet)
    tweet = re.sub(r'\'d', ' would', tweet)
    tweet = re.sub(r'\'t', ' not', tweet)
    tweet = re.sub(r'n\'t', ' not', tweet)
    
    return tweet

def remove_punctuations(tweet):
    return re.sub(r'[^\w\s]', '', tweet)

def remove_whitespaces(tweet):
    return str(tweet).strip()

def lemmatize(tweet):
    def __get_pos(tag):
        tags = {'N': wordnet.NOUN, 'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}

        try:
            return tags[tag]
        except KeyError:
            return wordnet.NOUN

    return ' '.join(lemmatizer.lemmatize(word[0], pos=__get_pos(word[1][0]))
                    for word in pos_tag(tweet.split()))

def remove_non_english_words(tweet):
    return ' '.join(word for word in tweet.split()
                    if word in words
                    and len(word) > 1)

def remove_english_stopwords(tweet):
    return ' '.join(word for word in tweet.split()
                    if word not in stopwords)

def preprocess_1(tweet):
    tweet = remove_links_and_hashtags(tweet)
    tweet = to_lowercase(tweet)
    tweet = remove_contractions(tweet)
    tweet = remove_punctuations(tweet)
    tweet = remove_whitespaces(tweet)

    return tweet

def preprocess_2(tweet):
    tweet = lemmatize(tweet)
    tweet = remove_non_english_words(tweet)
    tweet = remove_english_stopwords(tweet)

    return tweet

def remove_non_english_tweets(df):
    df['lang'] = df['cleanText'].apply(lambda text: detect(text))
    df = df.drop(df[df['lang'] != 'en'].index)
    df = df.drop(columns=['lang'])

    return df

def remove_empty_tweets(df):
    df = df.drop(df[df['cleanText'] == ''].index)

    return df

def preprocess(df):
    df['cleanText'] = df['cleanText'].apply(preprocess_1)
    df = remove_empty_tweets(df)
    df = remove_non_english_tweets(df)
    df['cleanText'] = df['cleanText'].apply(preprocess_2)
    df = remove_empty_tweets(df)

    return df
