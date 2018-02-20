"""Utilities to parse and analyse data
"""

import pandas as pd
import gzip
import numpy as np
import pickle as pickle
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def convertRowToReviewJson(row):
        json = {}
        json['reviewText'] = row['reviewText']
        json['helpful'] = row['helpful']
        json['summary'] = row['summary']
        return json


def filterReviews(columns):
        return columns['reviews'].tolist()[:2]


def preprocessReviewsDF(reviews_path):
    reviews_df = getDF('../data/reviews_electronics.json.gz')
    reviews_df['reviews'] = reviews_df[['reviewText', 'helpful', 'summary']].apply(convertRowToReviewJson, axis=1)

    reviews_filtered_df = reviews_df.groupby('asin').apply(filterReviews).reset_index()
    reviews_filtered_df.columns = ['asin', 'reviews']
    return reviews_filtered_df


def flatten(qa_reviews_df):
    qa_reviews_flattend_array = []

    for index, row in qa_reviews_df.iterrows():
        reviews = row['reviews']
        review_list = [review['reviewText'] for review in reviews]

        questions = row['questions']

        for question in questions:
            ans_list = []
            for answer in question['answers'][:2]:
                ans_list.append(answer['answerText'])
            item = [question['questionText'], review_list, ans_list]
            qa_reviews_flattend_array.append(item)

    return qa_reviews_flattend_array


def preprocess(qa_path, reviews_path):
    qa_df = getDF(qa_path)
    reviews_df = preprocessReviewsDF(reviews_path)

    qa_reviews_df = pd.merge(qa_df, reviews_df, on=['asin', 'asin'])

    qa_reviews_flattend_array = flatten(qa_reviews_df)
    pickle.dump(qa_reviews_flattend_array, open('qa_reviews_listed.pickle', 'wb'))
    #np.save('qa_reviews_listed', qa_reviews_flattend_array)


if __name__ == "__main__":
    category = 'Electronics'

    qa_path = '../data/QA_Electronics_multiple.json.gz'
    reviews_path = '../data/reviews_electronics.json.gz'

    preprocess(qa_path, reviews_path)

