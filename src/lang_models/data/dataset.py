import os
import torch
import string
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords

import constants as C
from data.vocabulary import Vocabulary
from data import review_utils
import string

DEBUG = False

class AmazonDataset(object):
    def __init__(self, params):
        self.model = params[C.MODEL_NAME]

        category = params[C.CATEGORY]

        self.max_question_len = params[C.MAX_QUESTION_LEN]
        self.max_answer_len = params[C.MAX_ANSWER_LEN]
        self.max_review_len = params[C.MAX_REVIEW_LEN]
        self.review_select_num = params[C.REVIEW_SELECT_NUM]
        self.review_select_mode = params[C.REVIEW_SELECT_MODE]

        self.max_vocab_size = params[C.VOCAB_SIZE]

        train_path = '%s/train-%s.pickle' % (C.INPUT_DATA_PATH, category)
        self.vocab = self.create_vocab(train_path)
        self.train = self.get_data(train_path)

        val_path = '%s/val-%s.pickle' % (C.INPUT_DATA_PATH, category)
        self.val = self.get_data(val_path)

        test_path = '%s/test-%s.pickle' % (C.INPUT_DATA_PATH, category)
        self.test = self.get_data(test_path)

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

    def truncate_tokens(self, text, max_length):
        return self.tokenize(text)[:max_length]

    def create_vocab(self, train_path):
        vocab = Vocabulary(self.max_vocab_size)
        assert os.path.exists(train_path)
        total_tokens = 0

        with open(train_path, 'rb') as f:
            dataFrame = pd.read_pickle(f)
            if DEBUG:
                dataFrame = dataFrame.iloc[:5]

        for _, row in dataFrame.iterrows():
            questionsList = row[C.QUESTIONS_LIST]
            for question in questionsList:
                tokens = self.truncate_tokens(question[C.TEXT], self.max_question_len)
                vocab.add_sequence(tokens)

                for answer in question[C.ANSWERS]:
                    tokens = self.truncate_tokens(answer[C.TEXT], self.max_answer_len)
                    total_tokens += len(tokens)
                    vocab.add_sequence(tokens)

            reviewsList = row[C.REVIEWS_LIST]
            for review in reviewsList:
                tokens = self.truncate_tokens(review[C.TEXT], self.max_review_len)
                vocab.add_sequence(tokens)

        print("Train: No. of Tokens = %d, Vocab Size = %d" % (total_tokens, vocab.size))
        return vocab


    def get_data(self, path):
        answersDict = []
        questionsDict = []
        reviewsDict = []
        questionAnswersDict = []

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
        for _, row in tqdm(dataFrame.iterrows()):
            tuples = []
            questionsList = row[C.QUESTIONS_LIST]

            reviewsDictList = []
            reviewsList = row[C.REVIEWS_LIST]
            review_tokens = []
            for review in reviewsList:
                tokens = self.tokenize(review[C.TEXT])
                review_tokens.append(tokens)
                ids = self.vocab.indices_from_token_list(tokens[:self.max_review_len])
                reviewsDict.append(ids)
                reviewId += 1
                reviewsDictList.append(reviewId)

            # Filter stop words and create inverted index
            review_tokens = [[token for token in r if token not in stop_words and token not in string.punctuation] for r in review_tokens]
            inverted_index = _create_inverted_index(review_tokens)
            review_tokens = list(map(set, review_tokens))

            # Add tuples to data 
            for question in questionsList:
                question_text = question[C.TEXT]
                question_tokens = self.tokenize(question_text)[:self.max_question_len]
                ids = self.vocab.indices_from_token_list(tokens)
                questionsDict.append(ids)
                questionId += 1

                answerIdsList = []
                if self.model == C.LM_QUESTION_ANSWERS_REVIEWS:
                    topReviewsDictList = review_utils.top_reviews(
                        set(question_tokens),
                        review_tokens,
                        inverted_index,
                        reviewsList,
                        reviewsDictList,
                        self.review_select_mode,
                        self.review_select_num
                    )

                for answer in question[C.ANSWERS]:
                    tokens = self.truncate_tokens(answer[C.TEXT], self.max_answer_len)
                    ids = self.vocab.indices_from_token_list(tokens)
                    answersDict.append(ids)
                    answerId += 1
                    answerIdsList.append(answerId)

                    if self.model == C.LM_ANSWERS:
                        tuples.append((answerId,))
                    elif self.model == C.LM_QUESTION_ANSWERS:
                        tuples.append((answerId, questionId))
                    elif self.model == C.LM_QUESTION_ANSWERS_REVIEWS:
                        tuples.append((answerId, questionId, topReviewsDictList))
                    else:
                        raise 'Unexpected'
                questionAnswersDict.append(answerIdsList)
            data.extend(tuples)

        assert(len(answersDict) == answerId+1)
        assert(len(questionsDict) == questionId+1)
        assert(len(reviewsDict) == reviewId+1)
        print("Number of samples in the data = %d" % (len(data)))

        return (answersDict, questionsDict, questionAnswersDict, reviewsDict, data)

def _create_inverted_index(review_tokens):
    term_dict = {}
    # TODO: Use actual review IDs
    for docId, tokens in enumerate(review_tokens):
        for token in tokens:
            if token in term_dict:
                if docId in term_dict[token]:
                    term_dict[token][docId] += 1
                else:
                    term_dict[token][docId] = 1
            else:
                term_dict[token] = {docId: 1}
    return term_dict
