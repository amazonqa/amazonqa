import numpy as np
import pandas as pd
import string
import pickle

import sklearn
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import sent_tokenize

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = 300

    def fit(self, X):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


def get_combined_review(top_reviews):
    all_reviews = ''
    for review in top_reviews:
        all_reviews += review.strip(' ')
        all_reviews += ' '
    return all_reviews.strip()

def tokenize(text):
    punctuations = string.punctuation.replace("\'", '')

    for ch in punctuations:
        text = text.replace(ch, " " + ch + " ")

    tokens = text.split()
    for i, token in enumerate(tokens):
        if not token.isupper():
            tokens[i] = token.lower()
    return tokens

def n_intersection(q, r):
    return len(set(q).intersection(set(r)))

def w2v_sim(q, r, w2v_vectorizer):
    # dot product of q and r as w2v vectors
    q_vec = w2v_vectorizer.transform([q])
    r_vec = w2v_vectorizer.transform([r])
    return q_vec.dot(r_vec.transpose())[0][0]

def tf_idf_sim(q, r, tfidf_vectorizer):
    # dot product of q and r as tfidf vectors
    q_vec = tfidf_vectorizer.transform([q])
    r_vec = tfidf_vectorizer.transform([r])
    return q_vec.dot(r_vec.transpose()).toarray()[0][0]

def tf_idf_sim_sentence(q, rs, tfidf_vectorizer):
    # max of dot products of q and each sentence in r as tfidf vectors
    q_vec = tfidf_vectorizer.transform([q])
    if len(rs) == 0:
        return 0
    return max([q_vec.dot(tfidf_vectorizer.transform([r]).transpose()).toarray()[0][0] for r in rs])

def w2v_sim_sentence(q, rs, w2v_vectorizer):
    # max of dot products of q and each sentence in r as tfidf vectors
    q_vec = w2v_vectorizer.transform([q])
    if len(rs) == 0:
        return 0
    return max([q_vec.dot(w2v_vectorizer.transform([r]).transpose())[0][0] for r in rs])

def tf_idf_sim_sentence_mean(q, rs, tfidf_vectorizer):
    # max of dot products of q and each sentence in r as tfidf vectors
    q_vec = tfidf_vectorizer.transform([q])
    if len(rs) == 0:
        return 0
    return np.mean([q_vec.dot(tfidf_vectorizer.transform([r]).transpose()).toarray()[0][0] for r in rs])

def w2v_sim_sentence_mean(q, rs, w2v_vectorizer):
    # max of dot products of q and each sentence in r as tfidf vectors
    q_vec = w2v_vectorizer.transform([q])
    if len(rs) == 0:
        return 0
    return np.mean([q_vec.dot(w2v_vectorizer.transform([r]).transpose())[0][0] for r in rs])

def load_classification_model(filename):

    with open(filename, 'rb') as fp:
        model = pickle.load(fp)

    return model

def load_vectorizers(tfidf_vectorizer_filename, w2v_vectorizer_filename):

    with open(tfidf_vectorizer_filename, 'rb') as fp:
        tfidf_vectorizer = pickle.load(fp)

    with open(w2v_vectorizer_filename, 'rb') as fp:
        w2v_vectorizer = pickle.load(fp)

    return {
        'w2v': w2v_vectorizer,
        'tfidf': tfidf_vectorizer
    }

def compute_features(vectorizers, question, reviews):
    q_tokens = tokenize(question)
    r_tokens = tokenize(reviews)
    r_sents = sent_tokenize(reviews)
    n_q = len(q_tokens)
    n_r = len(r_tokens)

    intersection_n = len(set(q_tokens).intersection(set(r_tokens)))
    intr_frac = intersection_n / n_q

    #tfidf = tf_idf_sim(question, reviews, vectorizers['tfidf'])
    #w2v = w2v_sim(question, reviews, vectorizers['w2v'])
    w2v_sent = w2v_sim_sentence(question, r_sents, vectorizers['w2v'])
    tfidf_sent = tf_idf_sim_sentence(question, r_sents, vectorizers['tfidf'])
    w2v_sent_mean = w2v_sim_sentence_mean(question, r_sents, vectorizers['w2v'])
    tfidf_sent_mean = tf_idf_sim_sentence_mean(question, r_sents, vectorizers['tfidf'])

    return np.array([
        [n_q, n_r, intersection_n, intr_frac, w2v_sent, tfidf_sent, w2v_sent_mean, tfidf_sent_mean]
    ])

def predictions_k(model, X, k):
    probs = model.predict_proba(X)
    return probs[:, 1] >= k

def is_answerable(model, vectorizers, question, top_reviews):
    features = compute_features(vectorizers, question, get_combined_review(top_reviews))
    return int(predictions_k(model, features, 0.7)[0])

def is_suggestive(model, vectorizers, question, top_reviews):
    features = compute_features(vectorizers, question, get_combined_review(top_reviews))
    return 1 - int(predictions_k(model, features, 0.5)[0])
