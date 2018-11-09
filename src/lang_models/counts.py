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
from evaluator.evaluator import COCOEvalCap
from operator import itemgetter, attrgetter
import json

DEBUG = False

class AmazonDataset(object):
    def __init__(self, params):
        self.review_select_num = params[C.REVIEW_SELECT_NUM]
        self.review_select_mode = params[C.REVIEW_SELECT_MODE]

        category = params[C.CATEGORY]
        self.train_path = '%s/train-%s.pickle' % (C.INPUT_DATA_PATH, category)
        self.val_path = '%s/val-%s.pickle' % (C.INPUT_DATA_PATH, category)
        self.test_path = '%s/test-%s.pickle' % (C.INPUT_DATA_PATH, category)

def main():
    model_name = C.LM_QUESTION_ANSWERS_REVIEWS
    params = config.get_model_params(model_name)
    params[C.MODEL_NAME] = model_name

    dataset = AmazonDataset(params)
    path = dataset.test_path

    assert os.path.exists(path)

    with open(path, 'rb') as f:
        dataFrame = pd.read_pickle(f)
        if DEBUG:
            dataFrame = dataFrame.iloc[:5]

    q_counts = []
    for (_, row) in dataFrame.iterrows():
        q_counts.append(len(row[C.QUESTIONS_LIST]))

    print(np.mean(q_counts), np.std(q_counts), len(q_counts))

if __name__ == '__main__':
    main()