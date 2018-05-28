import json
import re

import numpy as np
import operator
import constants as C

class BM25(object):
    def __init__(self, k1, k3, b):
        self.k1 = k1
        self.k3 = k3
        self.b = b

    def get_score(self, term_dict, common_tokens, N, docId, average_doc_length):
        score = 0
        doc_length = get_doc_length(term_dict, docId)
        for token in common_tokens:
            score += self.get_individual_term_score(term_dict, token, N, doc_length, average_doc_length)
        return score

    def get_individual_term_score(self, term_dict, token, N, doc_length, average_doc_length):
        df_term = len(term_dict[token])
        df_term = (N - df_term + 0.5) / (df_term + 0.5)
        idf = np.log(1 + df_term)

        tf = sum([term_dict[token][x] for x in term_dict[token].keys()])
        tf /= (tf + self.k1 * ((1 - self.b) + self.b * (doc_length / average_doc_length)))

        return tf * idf

class Indri(object):

    def __init__(self, lambda_, mu):
        self.lambda_ = lambda_
        self.mu = mu

    def get_score(self, inverted_index, question_tokens, doc_length, docId, N):
        score = 1.0
        for token in question_tokens:
            temp = self.get_individual_term_score(inverted_index, token, doc_length, docId, N)
            score *= temp

        score = np.power(score, float(1)/len(question_tokens))
        return score

    def get_individual_term_score(self, inverted_index, token, doc_length, docId, N):
        p_mle = 1.0 * max(np.sum(list(inverted_index.get(token, {}).values())), 1) / N
        score = self.lambda_ * p_mle
        score += (1 - self.lambda_) * (inverted_index.get(token, {}).get(docId, 0) + self.mu * p_mle)/ (doc_length + self.mu)
        return score

def get_average_sentence_length(index, N):
    total_terms = 0
    for i, term in enumerate(index):
        for j, doc in enumerate(index[term]):
            total_terms += index[term][doc]
    return float(total_terms) / N

def get_docID(snippet):
    doc_string = snippet['document']
    doc_id = int(doc_string.rstrip().replace('\'', '').split('/')[4])
    return doc_id

def update_dictionary(term_dict, tokens, docId):
    # updates the term dictionary for tokens of docId
    for token in tokens:
        if token in term_dict:
            if docId in term_dict[token]:
                term_dict[token][docId] += 1
            else:
                term_dict[token][docId] = 1
        else:
            term_dict[token] = {docId: 1}
    return term_dict

def get_doc_length(term_dict, docId):
    count = 0
    for key in term_dict:
        if docId in term_dict[key]:
            count += term_dict[key][docId]
    return count

def retrieval_model_scores(question_tokens, review_tokens, inverted_index, retrieval_algo):
    N = len(review_tokens)
    if N == 0:
        return []

    avg_length = get_average_sentence_length(inverted_index, N)

    bm25_model = BM25(k1=1.2, k3=0, b=0.75)
    indri_model = Indri(lambda_=0.75, mu=5000)
    scores = []

    for review_id, tokens in enumerate(review_tokens):
        if retrieval_algo == C.BM25:
            common_tokens = set(question_tokens).intersection(tokens)
            score = bm25_model.get_score(
                term_dict=inverted_index, 
                common_tokens=common_tokens, N=N,
                average_doc_length=avg_length, docId=review_id
            )
        elif retrieval_algo == C.INDRI:
            score = indri_model.get_score(
                inverted_index=inverted_index,
                question_tokens=question_tokens,
                docId=review_id, doc_length=len(tokens), N=N
            )
        else:
            raise 'Unimplemented retrieval algo: %s' % retrieval_algo
        scores.append(score)
    return scores
