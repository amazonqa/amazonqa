from torch.utils.data import Dataset, DataLoader
import pickle
from constants import *
import pandas as pd
import numpy as np
from tqdm import tqdm

class AnsDataset(Dataset):
    def __init__(self, category, batch_size):
        answers = []
        ansDict = Dictionary()
        qdata = list(pickle.load(open(JSON_DATA_PATH + '/' + \
            category + '.pickle', 'rb'))['questionsList'])

        for q in tqdm(qadata):
            for ans in q['answers']:
                answer = []
                anstext = ans['text']
                tokens = ['<sos>'] + line.split + ['<eos>']
                for token in tokens:
                    ansDict.add_word(word)
                    answer.append(ansDict.word2idx[word])
                answers.append(answer)
        answers.sort(key = lambda s: len(s))
        answers = np.array(answers)
        num_splits = int(np.ceil(len(x)/batch_size))
        batched_answers = np.array_split(answers, num_splits)
        np.random.shuffle(batched_answers)
        answers_flattened = np.hstack(batched_answers)

        self.dicts = [ansDict]
        self.datasets = [answers_flattened]

    def __getitem__(self, index):
        return self.datasets[0][index]

    def __len__(self):
        return len(self.datasets[0])


class Dictionary(object):
    def __init__(self):
        self.word2idx = {'<sos>':0, '<eos>':1, '<pad>':2, 3:'<unk>'}
        self.idx2word = {0:'<sos>', 1:'<eos>', 2:'<pad>', 3:'<unk>'}
        self.num_tokens = 4

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.num_tokens += 1

    def __len__(self):
        return len(self.word2idx)

def collate_fn(data):
    """
    takes batch_size of data and returns batch_size
    """
    #A = zip(*data)
    maxlen = np.max([len(a) for a in data])
    data = np.array( [ np.pad( ans, (0, maxlen - len(ans) ), mode='constant')  for ans in data] )
    np.random.shuffle(data)
    return data

def get_loader(typ, batch_size):
    ans_dataset = AnsDataset(typ, batch_size)
    sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    return dataloader

