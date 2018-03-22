import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import numpy as np
from operator import itemgetter, attrgetter
from torch.utils.data import Dataset, DataLoader


class AmazonDataLoader(DataLoader):

    def sortByLength(self, item):
        if self.mode is "1":
            return len(item)

        elif self.mode is "2":
            assert(len(item) == 2)
            return len(item[0])

        elif self.mode is "3":
            assert(len(item) == 3)
            reviews = item[2]
            max_len = 0
            for review in reviews:
                max_len = max(max_len, len(review))
            return max_len

    def __init__(self, data, mode, batch_size):
        self.batch_size = batch_size
        self.mode = mode

        sorted(data, key=self.sortByLength, reverse=True)
        self.data = data

    def create_packed_qa(self, batch_data):
        lengths = np.array([len(item) for item in batch_data])
        max_len = max(lengths)

        padded_data = np.array(
            [np.pad(item, (0, max_len - len(item)), 'constant') for item in batch_data])
        padded_data = torch.from_numpy(padded_data)

        return (padded_data)

    def create_packed_reviews(self, review_data):
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
        print(self.data)
        self.num_batches = len(self.data) // self.batch_size
        indices = np.arange(self.num_batches)
        np.random.shuffle(indices)

        for index in indices:
            start = index * self.batch_size
            end = (index + 1) * self.batch_size

            batch_data = self.data[start:end]
            assert(self.batch_size == len(batch_data))

            if self.mode is "1":
                answers = batch_data
                packed_answers = self.create_packed_qa(answers)
                yield (packed_answers)

            elif self.mode is "2":
                answers, questions = zip(*batch_data)
                packed_answers = self.create_packed_qa(list(answers))
                packed_questions = self.create_packed_qa(list(questions))
                yield (packed_answers, packed_questions)

            elif self.mode is "3":
                answers, questions, reviews = zip(*batch_data)
                packed_answers = self.create_packed_qa(list(answers))
                packed_questions = self.create_packed_qa(list(questions))
                packed_reviews = self.create_packed_reviews(list(reviews))
                yield (packed_answers, packed_questions, packed_reviews)

    def __len__(self):
        return self.num_batches
