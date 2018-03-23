import os
import torch
import constants as C
import pandas as pd
from vocabulary import Vocabulary
import string

class AmazonDataset(object):

    def __init__(self, category, mode, data_type):
        self.mode = mode
        self.vocab = Vocabulary(10000)

        self.topReviewsCount = 5

        self.reviewIds = []
        self.questionIds = []
        self.answerIds = []

        path = '%s/%s-%s.pickle' % (C.INPUT_DATA_PATH, data_type, category)
        self.data = self.get_data(path)


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
        assert os.path.exists(path)

        with open(path, 'rb') as f:
            data = pd.read_pickle(f)
            f.close()

        for index, row in data.iterrows():
            tuples = []
            questionsList = row[C.QUESTIONS_LIST]
            for question in questionsList:
                if C.TEXT in question:
                    text = question[C.TEXT]
                    self.vocab.add_sequence(self.tokenize(text))

                    for answer in question[C.ANSWERS]:
                        text = answer[C.TEXT]
                        self.vocab.add_sequence(self.tokenize(text))

            reviewsList = row[C.REVIEWS_LIST]
            for review in reviewsList:
                text = review[C.TEXT]
                self.vocab.add_sequence(self.tokenize(text))


        questionId = -1
        reviewId = -1
        answerId = -1

        final_data = []

        for index, row in data.iterrows():
            tuples = []
            questionsList = row[C.QUESTIONS_LIST]
            for question in questionsList:
                if C.TEXT in question:
                    text = question[C.TEXT]
                    ids = self.vocab.indices_from_sequence(self.tokenize(text))
                    self.questionIds.append(ids)
                    questionId += 1

                    for answer in question[C.ANSWERS]:
                        text = answer[C.TEXT]
                        ids = self.vocab.indices_from_sequence(self.tokenize(text))
                        self.answerIds.append(ids)
                        answerId += 1

                        if self.mode is "1":
                            tuples.append((answerId)) #check why zip(*a) doesnt work
                        else:
                            tuples.append((answerId, questionId))

            reviewsList = row[C.REVIEWS_LIST]
            reviewIdsList = []
            for review in reviewsList:
                text = review[C.TEXT]
                ids = self.vocab.indices_from_sequence(self.tokenize(text))
                self.reviewIds.append(ids)
                reviewId += 1

                if self.mode is "3":
                    if len(reviewIdsList) < self.topReviewsCount:
                        reviewIdsList.append(reviewId)

            if self.mode is "3":
                for i in range(len(tuples)):
                    tuples[i] = tuples[i] + (reviewIdsList,)

            final_data.extend(tuples)

        assert(len(self.answerIds) == answerId+1)
        assert(len(self.questionIds) == questionId+1)
        assert(len(self.reviewIds) == reviewId+1)

        return final_data
