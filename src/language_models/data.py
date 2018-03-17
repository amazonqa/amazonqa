import os
import torch
import constants as C
import pandas as pd


class Dictionary(object):
	def __init__(self):
		# word2idx - map word with index
		self.word2idx = {}

		#idx2word - get word given the index
		self.idx2word = []

	def add_word(self, word):
		if word not in self.word2idx:
			self.idx2word.append(word)
			self.word2idx[word] = len(self.idx2word)-1
		return self.word2idx[word]

	def __len__(self):
		return len(self.idx2word)


class AnswersCorpus(object):
	def __init__(self, category):
		self.dictionary = Dictionary()

		train_path = '%s/train-%s.pickle' % (C.INPUT_DATA_PATH, category)
		self.train = self.tokenize(train_path)

		valid_path = '%s/val-%s.pickle' % (C.INPUT_DATA_PATH, category)
		self.valid = self.tokenize(valid_path)

		test_path = '%s/test-%s.pickle' % (C.INPUT_DATA_PATH, category)
		self.test = self.tokenize(test_path)

	def tokenize(self, path):
		print(path)
		assert os.path.exists(path)

		with open(path, 'rb') as f:
			data = pd.read_pickle(f)

		tokens = 0
		for questionsList in data[C.QUESTIONS_LIST]:
			for question in questionsList:
				if 'answers' in question:
					for answer in question[C.ANSWERS]:
						line = answer[C.TEXT]
						words = ['<SOS>'] + line.split() + ['<EOS>']
						tokens += len(words)
						for word in words:
							self.dictionary.add_word(word)

		ids = torch.LongTensor(tokens)
		token = 0

		for questionsList in data[C.QUESTIONS_LIST]:
			for question in questionsList:
				if 'answers' in question:
					for answer in question[C.ANSWERS]:
						line = answer[C.TEXT]
						words = ['<SOS>'] + line.split() + ['<EOS>']
						for word in words:
							ids[token]= self.dictionary.word2idx[word]
							token += 1

		return ids
