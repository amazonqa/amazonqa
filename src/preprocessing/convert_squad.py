import os
import torch
import string
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords

import config
import constants as C
from data.vocabulary import Vocabulary
from data import review_utils
import string

DEBUG = False

class AmazonDataset(object):
    def __init__(self, params):
        self.review_select_num = params[C.REVIEW_SELECT_NUM]
        self.review_select_mode = params[C.REVIEW_SELECT_MODE]

        category = params[C.CATEGORY]
        self.train_path = '%s/train-%s.pickle' % (C.INPUT_DATA_PATH, category)
        self.val_path = '%s/val-%s.pickle' % (C.INPUT_DATA_PATH, category)
        self.test_path = '%s/test-%s.pickle' % (C.INPUT_DATA_PATH, category)

    def save_data(self, path, max_review_len=50, filename='temp.csv'):
        print("Creating Dataset from " + path)
        assert os.path.exists(path)

        with open(path, 'rb') as f:
            dataFrame = pd.read_pickle(f)
            if DEBUG:
                dataFrame = dataFrame.iloc[:5]

        print('Number of products: %d' % len(dataFrame))

        final_data = []
        for (_, row) in dataFrame.iterrows():
            # combine all or get only the reviews
            reviews = row[C.REVIEWS_LIST]

            qas = []
            for qid, question in enumerate(row[C.QUESTIONS_LIST]):
                question_text = question[C.TEXT]

                answers = question[C.ANSWERS]
                new_answers = find_span_answers(answers, reviews)
                
                is_answerable = find_answerable(question_text, reviews)
                new_question = {}
                new_question['id'] = qid
                new_question['is_impossible'] = is_answerable
                new_question['question'] = question_text
                new_question['answers'] = new_answers

                qas.append(new_question)
            
            new_row = {}
            new_row['context'] = reviews
            new_row['qas'] = qas
            final_data.append(new_row)
        
        pd.DataFrame(final_data).to_csv(filename)

def main():
    seed = 1
    max_review_len = 50
    np.random.seed(seed)
    model_name = C.LM_QUESTION_ANSWERS_REVIEWS
    params = config.get_model_params(model_name)
    params[C.MODEL_NAME] = model_name

    params[C.REVIEW_SELECT_MODE] = C.BM25
    dataset = AmazonDataset(params)
    path = dataset.test_path
    dataset.save_data(
        dataset.test_path,
        max_review_len=max_review_len,
        filename='squad_%s_%d_%d.csv' % (params[C.CATEGORY], max_review_len, seed)
    )

if __name__ == '__main__':
    main()
