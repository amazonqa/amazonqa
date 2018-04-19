"""Utilities for reviews
"""

import numpy.random as random
import constants as C
import math
import scipy.stats as st

def select_reviews(reviews, select_mode, num_reviews):
    if select_mode == C.RANDOM:
        random.shuffle(reviews)
    elif select_mode == C.WILSON:
        for r in range(len(reviews)):
            helpful_count = reviews[r]['helpful']
            # TODO: the 'unhelpful' key is wrong in the database! should be 'total'
            unhelpful_count = reviews[r]['unhelpful'] - helpful_count
            reviews[r]['wilson_score'] = wilson_score(helpful_count, unhelpful_count)
        reviews = sorted(reviews, key=lambda review: review['wilson_score'], reverse=True)
    else: #select_mode == HELPFUL or default 
        reviews = sorted(reviews, key=lambda review: review['helpful'], reverse=True)
    return reviews[:num_reviews]

def wilson_score(positive, negative):
    confidence = 0.98
    n = positive + negative
    if n == 0: return 0.
    phat = (1.*positive) / n
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    return (phat + z*z/(2*n) - z * math.sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n)
