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

    @staticmethod
    def tokenize(text):
        punctuations = string.punctuation.replace("\'", '')

        for ch in punctuations:
            text = text.replace(ch, " " + ch + " ")

        tokens = text.split()
        for i, token in enumerate(tokens):
            if not token.isupper():
                tokens[i] = token.lower()
        return tokens

    def save_data(self, path, num_entries, max_review_len=50, filename='temp.csv'):
        questionId = -1
        reviewId = -1
        answerId = -1
        data = []

        print("Creating Dataset from " + path)
        assert os.path.exists(path)

        with open(path, 'rb') as f:
            dataFrame = pd.read_pickle(f)
            if DEBUG:
                dataFrame = dataFrame.iloc[:5]

        if self.review_select_mode in [C.BM25, C.INDRI]:
            stop_words = set(stopwords.words('english'))
        else:
            stop_words = []

        print('Number of products: %d' % len(dataFrame))

        all_instances = []
        for row_id, (_, row) in enumerate(dataFrame.iterrows()):
            for qid, question in enumerate(row[C.QUESTIONS_LIST]):
                for aid, _ in enumerate(question[C.ANSWERS]):
                    all_instances.append((row_id, qid, aid))
        
        print('Total Instances: %d' % len(all_instances))
        print('Sampling %d instances ...' % num_entries)
        sample_ids = np.random.permutation(all_instances)[:num_entries]

        samples = []
        for ids in tqdm(sample_ids):
            row_id, qid, aid = ids
            row = dataFrame.iloc[row_id]
            question = row[C.QUESTIONS_LIST][qid]
            answer = question[C.ANSWERS][aid]

            reviews = row[C.REVIEWS_LIST]
           
            review_tokens = []
            review_texts = []
            for review in reviews:
                sentences = nltk.sent_tokenize(review[C.TEXT])
                bufr = []
                buffer_len = 0
                for sentence in sentences:
                    buffer_len += len(self.tokenize(sentence))
                    if buffer_len > max_review_len:
                        review_texts.append(' '.join(bufr))
                        bufr = []
                        buffer_len = 0
                    else:
                        bufr.append(sentence)

            # Filter stop words and create inverted index
            review_tokens = [self.tokenize(r) for r in review_texts]
            review_tokens = [[token for token in r if token not in stop_words and token not in string.punctuation] for r in review_tokens]
            inverted_index = _create_inverted_index(review_tokens)
            review_tokens = list(map(set, review_tokens))

            question_text = question[C.TEXT]
            answer_text = answer[C.TEXT]
            question_tokens = self.tokenize(question_text)

            top_reviews = review_utils.top_reviews(
                set(question_tokens),
                review_tokens,
                inverted_index,
                None,
                review_texts,
                self.review_select_mode,
                self.review_select_num
            )

            samples.append({
                'id': '(%d,%d,%d)' % tuple(ids),
                'question': question_text,
                'reviews': '\n'.join(top_reviews),
                'enumerated_reviews': '\n'.join(['%d) %s' % (rid + 1, r) for rid, r in enumerate(top_reviews) if r != '']),
                'answer': answer_text,
            })
        pd.DataFrame(samples)[['id', 'question', 'reviews', 'enumerated_reviews', 'answer']].to_csv(filename)

def _create_inverted_index(review_tokens):
    term_dict = {}
    # TODO: Use actual review IDs
    for doc_id, tokens in enumerate(review_tokens):
        for token in tokens:
            if token in term_dict:
                if doc_id in term_dict[token]:
                    term_dict[token][doc_id] += 1
                else:
                    term_dict[token][doc_id] = 1
            else:
                term_dict[token] = {doc_id: 1}
    return term_dict

if __name__ == '__main__':
    seed = 1
    np.random.seed(seed)
    model_name = C.LM_QUESTION_ANSWERS_REVIEWS    
    params = config.get_model_params(model_name)
    params[C.MODEL_NAME] = model_name

    params[C.REVIEW_SELECT_MODE] = C.BM25
    dataset = AmazonDataset(params)
    path = dataset.test_path
    dataset.save_data(
        dataset.test_path,
        100,
        max_review_len=50,
        filename='%s_samples_%d.csv' % (params[C.CATEGORY], seed)
    )
