from torch.utils.data import Dataset, DataLoader
import pickle
from constants import *
import pandas as pd
import numpy as np
from tqdm import tqdm


def get_data(category, split_type, typ):
    #TODO questions and reviews 
    return get_answers(category, split_type)


def get_answers(category, split_type):
    answers = []
    ansDict = Dictionary()
    qadata = list(pickle.load(open(INPUT_DATA_PATH + '/' + \
    split_type + '-' + category + '.pickle', 'rb'))['questionsList'])

    for q in tqdm(qadata):
        for ans in q:
            ans_ids = [] # my name is mansi # 13, 15, 12, 1
            anstext = ans['text']
            tokens = ['<sos>'] + anstext.split() + ['<eos>']
            for token in tokens:
                ansDict.add_word(token)
                ans_ids.append(ansDict.word2idx[token])
            answers.append(ans_ids)

    answers = np.array(answers)
    # TODO check sorted 
    indices = np.argsort([len(ans) for ans in answers])
    answers = answers[indices]
    return ansDict, answers


def get_batches(data):
    # should be called per epoch
    # data (np.array) should be sorted by length
    num_splits = int ( np.ceil ( len(data) / batch_size ) )
    batched_data = np.array_split( data, num_splits )
    batched_data = np.array(batched_data)
    np.random.shuffle(batched_data)
    # TODO shuffle inside batch
    #will pad by 0 '<pad>'
    maxlen = max([len(item) for item in data])
    padded_batched_data = np.array( [ np.pad( ass, (0, maxlen - len(ass) ), mode='constant')  for len in data] )
    return padded_batched_data #array( [], [], [] )


class Dictionary(object):
    # TODO: RARE WORDS HANDLING
    def __init__(self):
        self.word2idx = {'<pad>':0, '<sos>':1, '<eos>':2, '<unk>':3}
        self.idx2word = {0:'<pad>', 1:'<sos>', 2:'<eos>', 3:'<unk>'}
        self.num_tokens = 4

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.num_tokens
            self.idx2word[self.num_tokens] = word
            self.num_tokens += 1

    def __len__(self):
        return len(self.word2idx)
