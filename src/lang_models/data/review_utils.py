"""Utilities for reviews
"""

import numpy.random as random
import constants as C
import math
import scipy.stats as st
from data import retrieval_models

def top_reviews(question_tokens, review_tokens, inverted_index, reviews, review_ids, select_mode, num_reviews):
    _, top_review_ids = top_reviews_and_scores(question_tokens, review_tokens, inverted_index, reviews, review_ids, select_mode, num_reviews)
    return top_review_ids

def top_reviews_and_scores(question_tokens, review_tokens, inverted_index, reviews, review_ids, select_mode, num_reviews):
    if select_mode == C.RANDOM:
        scores = list(random.uniform(size=len(reviews)))
    elif select_mode in [C.BM25, C.INDRI]:
        scores = retrieval_models.retrieval_model_scores(question_tokens, review_tokens, inverted_index, select_mode)
    elif select_mode == C.WILSON:
        scores = []
        for r in range(len(reviews)):
            helpful_count = reviews[r]['helpful']
            # TODO: the 'unhelpful' key is wrong in the database! should be 'total'
            unhelpful_count = reviews[r]['unhelpful'] - helpful_count
            scores.append(_wilson_score(helpful_count, unhelpful_count))
    elif select_mode == C.HELPFUL:
        scores = [r['helpful'] for r in reviews]
    else:
        raise 'Unimplemented Review Select Mode'
    
    scores, top_review_ids = zip(*sorted(list(zip(scores, review_ids)), reverse=True)) if len(scores) > 0 else ([], [])
    return scores[:num_reviews], top_review_ids[:num_reviews]

def _top_review_ids(reviews, review_ids, sort_id):
    reviews_and_ids = [(r, rid) for r, rid in zip(reviews, review_ids)]
    top_reviews_and_ids = sorted(reviews_and_ids, key=lambda r, rid: r[sort_id], reverse=True)
    _, top_review_ids = zip(*top_reviews_and_ids)
    return top_review_ids

def _wilson_score(positive, negative):
    confidence = 0.98
    n = positive + negative
    if n == 0:
        return 0.0
    phat = (1.0 * positive) / n
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    return (phat + z*z/(2*n) - z * math.sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n)
