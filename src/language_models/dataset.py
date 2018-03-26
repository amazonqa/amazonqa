import os
import torch
import constants as C
import pandas as pd
from vocabulary import Vocabulary
import string

class AmazonDataset(object):

    def __init__(self, category, model, max_vocab_size):
        self.model = model

        self.topReviewsCount = 5
        self.max_vocab_size = max_vocab_size

        train_path = '%s/train-%s.pickle' % (C.INPUT_DATA_PATH, category)
        self.vocab = self.create_vocab(train_path)
        self.train = self.get_data(train_path)

        val_path = '%s/val-%s.pickle' % (C.INPUT_DATA_PATH, category)
        self.val = self.get_data(val_path)

        test_path = '%s/test-%s.pickle' % (C.INPUT_DATA_PATH, category)
        self.test = self.get_data(test_path)
        print("Dataset Loaded")


    def create_vocab(self, train_path):
        vocab = Vocabulary(C.MAX_VOCAB_SIZE)
        assert os.path.exists(train_path)

        with open(train_path, 'rb') as f:
            dataFrame = pd.read_pickle(f)
            f.close()

        for index, row in dataFrame.iterrows():
            tuples = []
            questionsList = row[C.QUESTIONS_LIST]
            for question in questionsList:
                if C.TEXT in question:
                    text = question[C.TEXT]
                    vocab.add_sequence(self.tokenize(text))

                    for answer in question[C.ANSWERS]:
                        text = answer[C.TEXT]
                        vocab.add_sequence(self.tokenize(text))

            reviewsList = row[C.REVIEWS_LIST]
            for review in reviewsList:
                text = review[C.TEXT]
                vocab.add_sequence(self.tokenize(text))
        return vocab


    def tokenize(self, text):
        punctuations = string.punctuation.replace("\'", '')

        for ch in punctuations:
            text = text.replace(ch, " " + ch + " ")

        tokens = text.split()

        for i in range(len(tokens)):
            token = tokens[i]
            if token.isupper() == False:
                tokens[i] = token.lower()
        return tokens


    def get_data(self, path):
        answersDict = []
        questionsDict = []
        reviewsDict = []

        questionId = -1
        reviewId = -1
        answerId = -1
        data = []

        assert os.path.exists(path)

        with open(path, 'rb') as f:
            dataFrame = pd.read_pickle(f)
            f.close()

        for index, row in dataFrame.iterrows():
            tuples = []
            questionsList = row[C.QUESTIONS_LIST]
            for question in questionsList:
                if C.TEXT in question:
                    text = C.SOS_TOKEN + question[C.TEXT] + C.EOS_TOKEN
                    ids = self.vocab.indices_from_token_list(self.tokenize(text))
                    questionsDict.append(ids)
                    questionId += 1

                    for answer in question[C.ANSWERS]:
                        text = C.SOS_TOKEN + answer[C.TEXT] + C.EOS_TOKEN
                        ids = self.vocab.indices_from_token_list(self.tokenize(text))
                        answersDict.append(ids)
                        answerId += 1

                        if self.model == C.LM_ANSWERS:
                            tuples.append((answerId,)) #check why zip(*a) doesnt work
                        else:
                            tuples.append((answerId, questionId))

            reviewsList = row[C.REVIEWS_LIST]
            reviewsDictList = []
            for review in reviewsList:
                text = C.SOS_TOKEN + review[C.TEXT] + C.EOS_TOKEN
                ids = self.vocab.indices_from_token_list(self.tokenize(text))
                reviewsDict.append(ids)
                reviewId += 1

                if self.model == C.LM_QUESTION_ANSWERS_REVIEWS:
                    if len(reviewsDictList) < self.topReviewsCount:
                        reviewsDictList.append(reviewId)

            if self.model == C.LM_QUESTION_ANSWERS_REVIEWS:
                for i in range(len(tuples)):
                    tuples[i] = tuples[i] + (reviewsDictList,)

            data.extend(tuples)

        assert(len(answersDict) == answerId+1)
        assert(len(questionsDict) == questionId+1)
        assert(len(reviewsDict) == reviewId+1)

        return (answersDict, questionsDict, reviewsDict, data)
