import os
import torch
import constants as C
import pandas as pd
from vocabulary import Vocabulary
import string

class AmazonDataset(object):

    def __init__(self, category, mode):
        self.mode = mode
        self.vocab = Vocabulary(10000)

        train_path = '%s/train-%s.pickle' % (C.INPUT_DATA_PATH, category)
        self.train = self.get_data(train_path)

        valid_path = '%s/val-%s.pickle' % (C.INPUT_DATA_PATH, category)
        #self.valid = self.get_data(valid_path)

        test_path = '%s/test-%s.pickle' % (C.INPUT_DATA_PATH, category)
        #self.test = self.get_data(test_path)


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

    # convert reviews row to json
    def reviewsToIds(self, row):
        reviewIdsList = []
        for review in row:
            reviewJson = {}
            text = C.SOS + review[C.TEXT] + C.EOS
            reviewJson[C.IDS] = self.vocab.indices_from_sequence(self.tokenize(text))
            reviewIdsList.append(reviewJson)
        return reviewIdsList


    def questionsToIds(self, questions_list):
        new_questions_list = []
        for question in questions_list:
            if C.TEXT in question:
                new_question = {}
                text = C.SOS + question[C.TEXT] + C.EOS
                new_question[C.IDS] = self.vocab.indices_from_sequence(
                    self.tokenize(text))

                new_answers = []
                for answer in question[C.ANSWERS]:
                    new_answer = {}
                    text = C.SOS + answer[C.TEXT] + C.EOS
                    new_answer[C.IDS] = self.vocab.indices_from_sequence(
                        self.tokenize(text))
                new_answers.append(new_answer)

                new_question[C.ANSWER_IDS_LIST] = new_answers
                new_questions_list.append(new_question)

        return new_questions_list


    def get_data(self, path):
        print(path)
        assert os.path.exists(path)

        with open(path, 'rb') as f:
            data = pd.read_pickle(f)

        for index, row in data.iterrows():
            questionsList = row['questionsList']
            for question in questionsList:
                if 'text' in question:
                    text = question['text']
                    self.vocab.add_sequence(self.tokenize(text))

                    for answer in question['answers']:
                        text = answer['text']
                        self.vocab.add_sequence(self.tokenize(text))

            reviewsList = row['reviewsList']
            for review in reviewsList:
                text = review['text']
                self.vocab.add_sequence(self.tokenize(text))

        data[C.REVIEW_IDS_LIST] = data[C.REVIEWS_LIST].apply(self.reviewsToIds)
        data[C.QUESTION_IDS_LIST] = data[C.QUESTIONS_LIST].apply(self.questionsToIds)

        final_data = []
        for index, row in data.iterrows():
            questionIdsList = row[C.QUESTION_IDS_LIST]
            for question in questionIdsList:
                tup = ()

                for answer in question[C.ANSWER_IDS_LIST]:
                    if self.mode is not "1":
                        ids = question[C.IDS]
                        tup += (ids,)

                    ids = answer[C.IDS]
                    if self.mode is "1":
                        final_data.append(ids)
                    else:
                        final_data.append(tup + (ids,))

            if self.mode is "3":
                reviewsList = row[C.REVIEW_IDS_LIST]
                reviewIds = []

                for review in reviewsList[0:2]:
                    ids = review[C.IDS]
                    reviewIds.append(ids)

                for i in range(len(final_data)):
                    final_data[i] += (reviewIds)

        return final_data
