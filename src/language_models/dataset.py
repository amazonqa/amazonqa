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

        self.topReviewsCount = 10

        self.reviews = []
        self.questions = []
        self.answers = []

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

        questionId = -1
        reviewId = -1
        answerId = -1

        final_data = []

        for index, row in data.iterrows():
            tuples = []
            questionsList = row['questionsList']
            for question in questionsList:
                if 'text' in question:
                    text = question['text']
                    self.vocab.add_sequence(self.tokenize(text))
                    self.questions.append(text)
                    questionId += 1

                    for answer in question['answers']:
                        text = answer['text']
                        self.vocab.add_sequence(self.tokenize(text))
                        self.answers.append(text)
                        answerId += 1

                        if self.mode is "1":
                            tuples.append((answerId)) #check why zip(*a) doesnt work
                        else:
                            tuples.append((answerId, questionId))

            reviewsList = row['reviewsList']
            reviewIds = []
            for review in reviewsList:
                text = review['text']
                self.vocab.add_sequence(self.tokenize(text))
                self.reviews.append(text)
                reviewId += 1

                if self.mode is "3":
                    if len(reviewIds) < self.topReviewsCount:
                        reviewIds.append(reviewId)

            if self.mode is "3":
                for i in range(len(tuples)):
                    tuples[i] = tuples[i] + (reviewIds,)

            final_data.extend(tuples)

        assert(len(self.answers) == answerId+1)
        assert(len(self.questions) == questionId+1)
        assert(len(self.reviews) == reviewId+1)

        return final_data
