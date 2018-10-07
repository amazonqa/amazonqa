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

    def find_answer_spans(answer_span_len, answers, reviews):
        for i in range(len(reviews)):
            span = review[i: i+answer_span_len]
            score = get_bleu(span, answers)
            l.append((score, span))

        sorted(l)
        return 

    def save_data(self, path, max_review_len=50, answer_span_len=10, filename='temp.csv'):
        print("Creating Dataset from " + path)
        assert os.path.exists(path)

        with open(path, 'rb') as f:
            dataFrame = pd.read_pickle(f)
            if DEBUG:
                dataFrame = dataFrame.iloc[:5]

        print('Number of products: %d' % len(dataFrame))

        paragraphs = []
        for (_, row) in dataFrame.iterrows():
            # combine all or get only the reviews
            all_reviews = processed(row[C.REVIEWS_LIST])
            qas = []
            for qid, question in enumerate(row[C.QUESTIONS_LIST]):
                context = get_context(question, all_reviews)
                relevant_context = get
                question_text = question[C.TEXT]

                answers = question[C.ANSWERS]
                new_answers = find_answer_spans(answer_span_len, answers, context)
                
                is_answerable = find_answerable(question_text, context)
                new_question = {}
                new_question['id'] = qid
                new_question['is_impossible'] = is_answerable
                new_question['question'] = question_text
                new_question['answers'] = new_answers

                qas.append(new_question)
            
            new_row = {}
            new_row['context'] = context
            new_row['qas'] = qas
            paragraphs.append(new_row)
        
        data = []
        data['title'] = 'AmazonDataset'
        data['paragraphs'] = paragraphs
        json.dumps(data, filename)

def main():
    seed = 1
    max_review_len = 50
    answer_span_len = 10
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
        answer_span_len=answer_span_len,
        filename='squad_%s_%d_%d.csv' % (params[C.CATEGORY], max_review_len, seed)
    )

if __name__ == '__main__':
    main()
