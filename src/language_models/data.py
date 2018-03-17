import os
import torch
import constants as C
import pandas as pd


class Dicitionary(object):
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
		self.dicitionary = Dicitionary()

		train_path = '%s/train-%s.pickle' % (C.INPUT_DATA_PATH, category)
		self.train = self.tokenize(train_path)

		val_path = '%s/val-%s.pickle' % (C.INPUT_DATA_PATH, category)
		self.val = self.tokenize(val_path)

		test_path = '%s/test-%s.pickle' % (C.INPUT_DATA_PATH, category)
		self.test = self.tokenize(test_path)

	def tokenize(self, path):
		assert os.path.exists(path)

		with open(path, 'rb') as f:
			data = pd.read_pickle(f)

		for questionsList in data[C.QUESTIONS_LIST]:
			for question in questionsList:
				for answer in question[C.ANSWERS]:
					line = answer[C.TEXT]
					words = '<SOS>' + line.split() + '<EOS>'
					for word in words:
						self.dicitionary.add_word(word)

		answerIds_List = []
		for questionsList in data[C.QUESTIONS_LIST]:
			for question in questionsList:
				for answer in question[C.ANSWERS]:
					line = answer[C.TEXT]
					words = '<SOS>' + line.split() + '<EOS>'
					answerIds = []
					for word in words:
						answerIds.append(self.dicitionary.word2idx[word])
					answerIds_List.append(np.array(answerIds))

		return np.array(answerIds_List)


class ShuffleDataLoader(DataLoader):
	def __init__(self, array, batch_size):
		random.shuffle(array)
		data = np.concatenate((array))
		m = len(data) // batch_size

		data = data[: m*batch_size+1]
		self.inputs = data[:-1].reshape(batch_size, m).T
		self.labels = data[1:].reshape(batch_size, m).T

		self.inputs = torch.from_numpy(self.inputs).long()
		self.labels = torch.from_numpy(self.labels).long()

		if torch.cuda.is_available():
			self.inputs = self.inputs.cuda()
			self.labels = self.labels.cuda()

	def __iter__(self):
		for i in range(self.len):
			start = i*self.seq_length
			end = (i+1)*self.seq_length

		yield (self.inputs[start:end], self.labels[start:end])

	def __len__(self):
		self.seq_length = 100
		self.len = self.inputs.shape[0] // self.seq_length
		return self.len
