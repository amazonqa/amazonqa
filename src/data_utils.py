"""Utilities to parse and analyse data
"""

import pickle
import pandas as pd
import gzip
import numpy as np

AUTOMOTIVE = 'Automotive'
BABY = 'Baby'
BEAUTY = 'Beauty'
CELL_PHONES_AND_ACCESSORIES = 'Cell_Phones_and_Accessories'
CLOTHING_SHOES_AND_JEWELRY = 'Clothing_Shoes_and_Jewelry'
ELECTRONICS = 'Electronics'
GROCERY_AND_GOURMET_FOOD = 'Grocery_and_Gourmet_Food'
HEALTH_AND_PERSONAL_CARE = 'Health_and_Personal_Care'
HOME_AND_KITCHEN = 'Home_and_Kitchen'
MUSICAL_INSTRUMENTS = 'Musical_Instruments'
OFFICE_PRODUCTS = 'Office_Products'
PATIO_LAWN_AND_GARDEN = 'Patio_Lawn_and_Garden'
PET_SUPPLIES = 'Pet_Supplies'
SPORTS_AND_OUTDOORS = 'Sports_and_Outdoors'
TOOLS_AND_HOME_IMPROVEMENT = 'Tools_and_Home_Improvement'
TOYS_AND_GAMES = 'Toys_and_Games'
VIDEO_GAMES = 'Video_Games'

CATEGORIES = [
  AUTOMOTIVE,
  BABY,
  BEAUTY,
  CELL_PHONES_AND_ACCESSORIES,
  CLOTHING_SHOES_AND_JEWELRY,
  ELECTRONICS,
  GROCERY_AND_GOURMET_FOOD,
  HEALTH_AND_PERSONAL_CARE,
  HOME_AND_KITCHEN,
  MUSICAL_INSTRUMENTS,
  OFFICE_PRODUCTS,
  PATIO_LAWN_AND_GARDEN,
  PET_SUPPLIES,
  SPORTS_AND_OUTDOORS,
  TOOLS_AND_HOME_IMPROVEMENT,
  TOYS_AND_GAMES,
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

def save_tables_for_all_categories():
    """Processes .tar.gz files for all the categories
    and saves qa & reviews tables in a .pickle format
    """
    for category in CATEGORIES:
        create_qa_review_tables(category)
