"""Utilities to parse and analyse data
"""

import pickle
import pandas as pd
import gzip
import numpy as np
import nltk
from nltk.corpus import stopwords
import random

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

DATA_PATH = '../data/pickle_files'
NN_DATA_PATH = '../data/nn/'

def filepath(category, key):
    if key == QA:
        path = '%s/QA_%s.json.gz' % (DATA_PATH, category)
    elif key == REVIEWS:
        path = '%s/reviews_%s_5.json.gz' % (DATA_PATH, category)
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

    with open('%s/%s.pickle' % (DATA_PATH, category), 'wb') as f:
        pickle.dump((qa_table, reviews), f)

def tables_from_category(category):
    with open('%s/%s.pickle' % (DATA_PATH, category), 'rb') as f:
        return pickle.load(f, encoding='latin1')

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

def token_count(line):
    return len(line.split(' '))

def get_category_stats(category):
    qa_table, reviews_table = tables_from_category(category)

    # Number of Products
    num_products = len(np.unique(qa_table.asin))

    # Number of questions
    question_table = qa_table[['asin', 'questionText', 'questionType']].drop_duplicates()
    num_questions = len(question_table)
    vc = question_table['questionType'].value_counts()
    num_yes_no, num_oe = vc['yes/no'], vc['open-ended']

    # Number of answers
    num_answers = len(qa_table)

    # Number of reviews
    num_reviews = len(reviews_table)

    # Per product
    avg_reviews = reviews_table.groupby(['asin']).count()['reviewText'].mean()

    # Avg question per product & answers per question
    avg_questions_per_product = qa_table[['asin', 'questionText']].drop_duplicates().groupby('asin')['questionText'].count().mean()
    avg_answers_per_question = qa_table.groupby(['questionText']).count()['answerText'].mean()

    # Duplicate questions
    df = question_table.groupby(['questionText']).count()
    num_dupl_questions = len(df[df.asin > 1])

    return {
        'Num Products': num_products,
        'Num Questions': num_questions,
        'Num Answers': num_answers,
        'Num Yes/No Questions': num_yes_no,
        'Num Open Ended Questions': num_oe,
        'Num Reviews': num_reviews,
        'Avg Reviews Per Product': avg_reviews,
        'Avg Questions Per Product': avg_questions_per_product,
        'Avg Answers Per Question': avg_answers_per_question,
        'Num Duplicate Questions': num_dupl_questions,
    }

def data_stats():
    rows = []
    categories = CATEGORIES
    for category in categories:
        rows.append(get_category_stats(category))
    return pd.DataFrame(rows, columns=[
        'Num Products',
        'Num Questions',
        'Num Answers',
        'Num Yes/No Questions',
        'Num Open Ended Questions',
        'Num Reviews',
        'Avg Reviews Per Product',
        'Avg Questions Per Product',
        'Avg Answers Per Question',
        'Num Duplicate Questions',
    ], index=categories)

# get question with the most helpful review and random answer
def save_qa_pairs_train_test(category, train_ratio):
    qa_table, reviews_table = tables_from_category(category)

    fn = lambda obj: obj.loc[np.random.choice(obj.index, 1, False),:]
    fqa_helpful = qa_table[['asin', 'questionText', 'answerText']]\
            .drop_duplicates().groupby('questionText', as_index=False)\
            .apply(fn).reset_index()[['asin', 'questionText', 'answerText']]
    print("Processed QA")

    rev = reviews_table[['asin', 'helpful', 'reviewText']]
    qa_rev = pd.merge(rev, fqa_helpful, how='inner', on=['asin', 'asin'])
    qa_rev['helpful'] = qa_rev['helpful'].apply(lambda l : l[0])

    qa_rev_helpful = qa_rev.sort_values('helpful', ascending=False)\
            .groupby(['questionText', 'answerText'], as_index=False).first()

    #df = qa_rev_helpful[['questionText', 'answerText', 'reviewText']]

    print("Processed Reviews")

    qardata = list(zip( list(qa_rev_helpful['questionText']), \
                   list(qa_rev_helpful['answerText']), \
                   list(qa_rev_helpful['reviewText'])
                   ))
    for item in qardata[0]:
        print(item + '\n\n')
    print(len(qardata))

    index = int(len(qardata) * train_ratio)
    qar_train = qardata[0:index]
    qar_test = qardata[index:]
    pickle.dump(qar_train, open(NN_DATA_PATH + category + '_qar_train.pickle', 'wb'))
    pickle.dump(qar_test, open(NN_DATA_PATH + category + '_qar_test.pickle', 'wb'))

    print("Saved data")

    return qardata

def get_review_text(category, n):
    qa_table, reviews_table = tables_from_category(category)
    reviews = list(reviews_table['reviewText'].unique())

    stopset = set(stopwords.words('english'))

    ngrams = []

    sample_size = len(reviews)
    random_reviews = [reviews[i] for i in random.sample(range(0, len(reviews)), sample_size)]

    for review in random_reviews:
        tokens = nltk.word_tokenize(review)
        tokens = [w for w in tokens if not w in stopset]
        ngrams.extend(list(nltk.ngrams(tokens, n)))

    final_list = []

    fdist = nltk.FreqDist(ngrams)
    for word, frequency in fdist.items():
        final_list.append([" ".join(word), frequency])

    ngram_freq_array = np.array(final_list)
    with open('%s/%d-gram-%s.pickle' % (DATA_PATH, n, category), 'wb') as f:
        pickle.dump(ngram_freq_array, f)

    file = open(str(n)+'-grams-top100.txt', 'w')

    for word, frequency in fdist.most_common(100):
        file.write(" ".join(word)+' '+str(frequency)+'\n')

    return ngram_freq_array

#save_qa_pairs_train_test(ELECTRONICS, 0.8)
#save_qa_pairs_train_test(TOYS_AND_GAMES, 0.8)
