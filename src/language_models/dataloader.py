import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import numpy as np
from operator import itemgetter, attrgetter
from torch.utils.data import Dataset, DataLoader


class AmazonDataLoader(object):

    def sortByLength(self, item):
        if self.mode is "1":
            answer = self.dataset.answers[item]
            return len(answer)

        elif self.mode is "2":
            assert(len(item) == 2)
            answer = self.dataset.answers[item[0]]
            return len(answer)

        elif self.mode is "3":
            assert(len(item) == 3)
            reviewIds = item[2]
            max_len = 0
            for reviewId in reviewIds:
                review = self.dataset.reviews[reviewId]
                max_len = max(max_len, len(review))
            return max_len


    def __init__(self, dataset, mode, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.mode = mode

        self.dataset.data = sorted(self.dataset.data, key=self.sortByLength, reverse=True)


    def pad_answers(self, answerIds):
        batch_data = []
        for answerId in answerIds:
            tokens = self.dataset.tokenize(self.dataset.answers[answerId])
            ids = self.dataset.vocab.indices_from_sequence(tokens)
            batch_data.append(ids)

        lengths = np.array([len(item) for item in batch_data])
        max_len = max(lengths)

        padded_data = np.array(
            [np.pad(item, (0, max_len - len(item)), 'constant') for item in batch_data])
        padded_data = torch.from_numpy(padded_data)

        return (padded_data)


    def pad_questions(self, questionIds):
        batch_data = []
        for questionId in questionIds:
            tokens = self.dataset.tokenize(self.dataset.questions[questionId])
            ids = self.dataset.vocab.indices_from_sequence(tokens)
            batch_data.append(ids)

        lengths = np.array([len(item) for item in batch_data])
        max_len = max(lengths)

        padded_data = np.array(
            [np.pad(item, (0, max_len - len(item)), 'constant') for item in batch_data])
        padded_data = torch.from_numpy(padded_data)

        return (padded_data)

    def pad_reviews(self, reviewIdsList):
        review_data = []
        for reviewIds in reviewIdsList:
            reviews = []
            for reviewId in reviewIds:
                tokens = self.dataset.tokenize(self.dataset.reviews[reviewId])
                ids = self.dataset.vocab.indices_from_sequence(tokens)
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
            padded_batch_data = torch.from_numpy(padded_batch_data)

            padded_data.append(padded_batch_data)

        return padded_data


    def __iter__(self):
        self.num_batches = len(self.dataset.data) // self.batch_size
        indices = np.arange(self.num_batches)
        np.random.shuffle(indices)

        for index in indices:
            start = index * self.batch_size
            end = (index + 1) * self.batch_size

            batch_data = self.dataset.data[start:end]
            assert(self.batch_size == len(batch_data))

            if self.mode is "1":
                answerIds = batch_data
                paded_answers = self.pad_answers(answerIds)
                yield (paded_answers)

            elif self.mode is "2":
                answerIds, questionIds = zip(*batch_data)
                paded_answers = self.pad_answers(list(answerIds))
                padded_questions = self.pad_questions(list(questionIds))
                yield (paded_answers, padded_questions)

            elif self.mode is "3":
                answerIds, questionIds, reviewIds = zip(*batch_data)
                paded_answers = self.pad_answers(list(answerIds))
                padded_questions = self.pad_questions(list(questionIds))
                padded_reviews = self.pad_reviews(list(reviewIds))
                yield (paded_answers, padded_questions, padded_reviews)


    def __len__(self):
        return self.num_batches
