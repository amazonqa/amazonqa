import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import numpy as np
from operator import itemgetter, attrgetter
from torch.utils.data import Dataset, DataLoader

import constants as C

class AmazonDataLoader(object):

    def __init__(self, data, model, batch_size):
        self.answersDict, self.questionsDict, self.questionAnswersDict, self.reviewsDict, self.data = data

        self.batch_size = batch_size
        self.model = model
        self.data = sorted(self.data, key=self.sortByLength, reverse=True)
        self.num_batches = len(self.data) // self.batch_size

    def sortByLength(self, item):
        max_len = 0

        if self.model == C.LM_ANSWERS:
            assert(len(item) == 1)
            answer = self.answersDict[item[0]]
            max_len = len(answer)

        elif self.model == C.LM_QUESTION_ANSWERS:
            assert(len(item) == 2)
            question = self.questionsDict[item[1]]
            max_len = len(question)

        elif self.model == C.LM_QUESTION_ANSWERS_REVIEWS:
            assert(len(item) == 3)
            #reviewIds = item[2]
            #for reviewId in reviewIds:
            #   review = self.reviewsDict[reviewId]
            #   max_len = max(max_len, len(review))
            question = self.questionsDict[item[1]]
            max_len = len(question)
        else:
            raise 'Unknown Model %s' % self.model

        return max_len

    def pad_answers(self, answerIds):
        batch_data = []
        for answerId in answerIds:
            ids = self.answersDict[answerId]
            batch_data.append(ids)

        lengths = np.array([len(item) for item in batch_data])
        max_len = max(lengths)

        padded_data = np.array(
            [np.pad(item, (0, max_len - len(item)), 'constant') for item in batch_data])
        return (padded_data, lengths)


    def pad_questions(self, questionIds):
        batch_data = []
        for questionId in questionIds:
            ids = self.questionsDict[questionId]
            batch_data.append(ids)

        lengths = np.array([len(item) for item in batch_data])
        max_len = max(lengths)

        padded_data = np.array(
            [np.pad(item, (0, max_len - len(item)), 'constant') for item in batch_data])
        padded_data = _reverse(padded_data)

        return (padded_data)

    def pad_reviews(self, reviewIdsList):
        review_data = []
        for reviewIds in reviewIdsList:
            reviews = []
            for reviewId in reviewIds:
                ids = self.reviewsDict[reviewId]
                reviews.append(ids)
            review_data.append(reviews)

        max_num_reviews = 0
        for reviews in review_data:
            max_num_reviews = max(max_num_reviews, len(reviews))

        data = []
        for i in range(max_num_reviews):
            batch_data = []
            for j in range(self.batch_size):
                reviews = review_data[j]
                if i < len(reviews):
                    batch_data.append(reviews[i])
                else:
                    batch_data.append([0])
            data.append(batch_data)

        padded_data = []
        for i in range(max_num_reviews):
            batch_data = data[i]
            lengths = [len(review) for review in batch_data]
            max_len = max(lengths)

            padded_batch_data = np.array(
                [np.pad(item, (0, max_len - len(item)), 'constant') for item in batch_data])
            padded_batch_data = _reverse(padded_batch_data)

            padded_data.append(padded_batch_data)

        return padded_data


    def __iter__(self):
        indices = np.arange(self.num_batches)
        np.random.shuffle(indices)

        for index in indices:
            start = index * self.batch_size
            end = (index + 1) * self.batch_size

            batch_data = self.data[start:end]
            #print(batch_data)
            assert(self.batch_size == len(batch_data))

            if self.model == C.LM_ANSWERS:
                [answerIds] = zip(*batch_data)
                paded_answers = self.pad_answers(list(answerIds))
                yield (paded_answers)

            elif self.model == C.LM_QUESTION_ANSWERS:
                [answerIds, questionIds] = zip(*batch_data)
                paded_answers = self.pad_answers(list(answerIds))
                padded_questions = self.pad_questions(list(questionIds))
                yield (paded_answers, padded_questions, list(questionIds))

            elif self.model == C.LM_QUESTION_ANSWERS_REVIEWS:
                [answerIds, questionIds, reviewIds] = zip(*batch_data)
                paded_answers = self.pad_answers(list(answerIds))
                padded_questions = self.pad_questions(list(questionIds))
                padded_reviews = self.pad_reviews(list(reviewIds))
                yield (paded_answers, padded_questions, list(questionIds), padded_reviews)


    def __len__(self):
        return self.num_batches

def _reverse(array):
    return np.ascontiguousarray(array[:, ::-1])
