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
import json

DEBUG = False

class AmazonDataset(object):
    def __init__(self, params, mode):
        self.model = params[C.MODEL_NAME]
        self.max_question_len = params[C.MAX_QUESTION_LEN]
        self.max_answer_len = params[C.MAX_ANSWER_LEN]
        self.max_review_len = params[C.MAX_REVIEW_LEN]
        self.review_select_num = params[C.REVIEW_SELECT_NUM]
        self.review_select_mode = params[C.REVIEW_SELECT_MODE]
        self.max_vocab_size = params[C.VOCAB_SIZE]
        suffix = 'qar_all'

        train_path = '%s/train-%s.jsonl' % (C.INPUT_DATA_PATH, suffix)
        self.vocab = self.create_vocab(train_path)

        if mode == C.TRAIN_TYPE:
            self.train = self.get_data(train_path)

            val_path = '%s/val-%s.jsonl' % (C.INPUT_DATA_PATH, suffix)
            self.val = self.get_data(val_path)

        if mode == C.DEV_TYPE or mode == C.TEST_TYPE:
            test_path = '%s/test-%s.jsonl' % (C.INPUT_DATA_PATH, suffix)
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

        with open(train_path, 'r') as fp:
            for line in fp:
                try:
                    question = json.loads(line)
                except json.JSONDecodeError:
                    raise Exception('\"%s\" is not a valid json' % line)

                if question[C.IS_ANSWERABLE] == 0:
                    continue

                tokens = self.truncate_tokens(question[C.QUESTION_TEXT], self.max_question_len)
                vocab.add_sequence(tokens)

                for answer in question[C.ANSWERS]:
                    tokens = self.truncate_tokens(answer[C.ANSWER_TEXT], self.max_answer_len)
                    total_tokens += len(tokens)
                    vocab.add_sequence(tokens)

                for review in question[C.REVIEW_SNIPPETS]:
                    tokens = self.truncate_tokens(review, self.max_review_len)
                    vocab.add_sequence(tokens)

        print("Train: No. of Tokens = %d, Vocab Size = %d" % (total_tokens, vocab.size))
        return vocab

    def get_data(self, path):
        answersDict = []
        questionsDict = []
        reviewsDict = []
        questionAnswersDict = []
        reviewsDictList = []

        questionId = -1
        reviewId = -1
        answerId = -1
        data = []

        print("Creating Dataset from " + path)
        assert os.path.exists(path)

        with open(path, 'r') as fp:
            for line in fp:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    raise Exception('\"%s\" is not a valid json' % line)

                if row[C.IS_ANSWERABLE] == 0:
                    continue

                question = row[C.QUESTION_TEXT]
                answers = row[C.ANSWERS]
                reviews = row[C.REVIEW_SNIPPETS][:5]    # TODO: Make this limit (5) a parameter

                review_tokens = []
                reviewsDictList = []
                for review in reviews:
                    tokens = self.tokenize(review)
                    review_tokens.append(tokens)
                    review_ids = self.vocab.indices_from_token_list(tokens[:self.max_review_len])
                    reviewsDict.append(review_ids)
                    reviewId += 1
                    reviewsDictList.append(reviewId)

                # Add tuples to data 
                question_tokens = self.tokenize(question)[:self.max_question_len]
                question_ids = self.vocab.indices_from_token_list(question_tokens)
                questionsDict.append(question_ids)
                questionId += 1

                answerIdsList = []
                tuples = []
                for answer in answers:
                    answer_tokens = self.truncate_tokens(answer[C.ANSWER_TEXT], self.max_answer_len)
                    answer_ids = self.vocab.indices_from_token_list(answer_tokens)
                    answersDict.append(answer_ids)
                    answerId += 1
                    answerIdsList.append(answerId)

                    if self.model == C.LM_ANSWERS:
                        tuples.append((answerId,))
                    elif self.model == C.LM_QUESTION_ANSWERS:
                        tuples.append((answerId, questionId))
                    elif self.model == C.LM_QUESTION_ANSWERS_REVIEWS:
                        tuples.append((answerId, questionId, reviewsDictList))
                    else:
                        raise 'Unexpected'
                questionAnswersDict.append(answerIdsList)
                data.extend(tuples)

        assert(len(answersDict) == answerId+1)
        assert(len(questionsDict) == questionId+1)
        assert(len(reviewsDict) == reviewId+1)
        data_size = len(data)
        print("Number of samples in the data = %d" % (data_size))

        return (answersDict, questionsDict, questionAnswersDict, reviewsDict, data)
