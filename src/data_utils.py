"""Utilities to parse and analyse data
"""

import pickle
import pandas as pd
import gzip
import numpy as np

VIDEO_GAMES = 'Video_Games'

CATEGORIES = [
  VIDEO_GAMES,
]

QA = 'QA'
REVIEWS = 'Reviews'

def filepath(category, key):
    if key == QA:
        path = '../data/answers_multiple/QA_%s.json.gz' % category
    elif key == REVIEWS:
        path = '../data/reviews_small/reviews_%s_5.json.gz' % category
    else:
        raise 'Unexpected key'
    return path

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

def create_qa_review_tables(category):
    qa, reviews = [getDF(filepath(category, key)) for key in [QA, REVIEWS]]
    common_products = set(qa.asin.values).intersection(reviews.asin.values)
    qa_rows = []

    for _, qa_row in qa.iterrows():
        if qa_row.asin in common_products:
            for question in qa_row.questions:
                for answer in question['answers']:
                    row = {'asin': qa_row.asin}
                    for key in ['questionText', 'askerID', 'questionType']:
                        row[key] = question.get(key, None)
                    for key in ['answererID', 'answerType', 'answerText', 'answerScore']:
                        row[key] = answer.get(key, None)
                    qa_rows.append(row)
    qa_table = pd.DataFrame(qa_rows)
    reviews = reviews[reviews.asin.isin(common_products)]

    with open('../data/%s.pickle' % category, 'wb') as f:
        pickle.dump((qa_table, reviews), f)

def tables_from_category(category):
    with open('../data/%s.pickle' % category, 'rb') as f:
        return pickle.load(f)

def convertRowToReviewJson(row):
    json = {}
    json['reviewText'] = row['reviewText']
    json['helpful'] = row['helpful']
    json['summary'] = row['summary']
    return json

def filterReviews(columns):
    return columns['reviews'].tolist()[:2]

def preprocessReviewsDF(reviews_path):
    reviews_df = getDF('../data/reviews_'+category+'_5.json.gz')
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
      for answer in question['answers'][:2]:
        item = (review_list, question['questionText'], answer['answerText'])
        qa_reviews_flattend_array.append(item)

  return np.array(qa_reviews_flattend_array)

def save_tables_for_all_categories():
    """Processes .tar.gz files for all the categories
    and saves qa & reviews tables in a .pickle format
    """
    for category in CATEGORIES:
        create_qa_review_tables(category)

def preprocess(qa_path, reviews_path):
    qa_df = getDF(qa_path)
    reviews_df = preprocessReviewsDF(reviews_path)

    qa_reviews_df = pd.merge(qa_df, reviews_df, on=['asin', 'asin'])

    qa_reviews_flattend_array = flatten(qa_reviews_df)
    np.save('qa_reviews', qa_reviews_flattend_array)

# if __name__ == "__main__":
#   category = 'Video_Games'

#   qa_path = '../data/QA_%s_multiple.json.gz' % (category)
#   reviews_path = '../data/reviews_%s_5.json.gz' % (category)

#   preprocess(qa_path, reviews_path)
