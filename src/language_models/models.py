import torch.nn as nn
from torch.autograd import Variable


class RNNModel(nn.Module):
	def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout=0.5, tie_weights=False):
		super(RNNModel, self).__init__()

		self.drop = nn.Dropout(dropout)
		self.encoder = nn.Embedding(vocab_size, embedding_size)
		self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout = dropout)
		self.decoder = nn.Linear(hidden_size, vocab_size)

		if tie_weights:
			assert (embedding_size == hidden_size)
			self.encoder.weight = self.decoder.weight

	def forward(self, inputs):
		out = inputs

		out = self.drop(self.encoder(out))
		out, states = self.rnn(out)
		out = self.drop(self.decoder(out))
		return out


class RLSTMModel(nn.Module):
	def __init__(self, vocab_size, embedding_size, hidden_size):
		super(RLSTMModel, self).__init__()

		self.encoder = nn.Embedding(vocab_size, embedding_size)
		self.rnns = nn.ModuleList([
			nn.LSTM(input_size=embedding_size, hidden_size=hidden_size),
			nn.LSTM(input_size=hidden_size, hidden_size=hidden_size),
			nn.LSTM(input_size=hidden_size, hidden_size=embedding_size)
		])
		self.decoder = nn.Linear(embedding_size, vocab_size)

		self.encoder.weight = self.decoder.weight

	def forward(self, inputs):
		out = inputs

		out = self.encoder(out)
		for rnn in self.rnns:
			out, states = rnn(out)
		out = self.decoder(out)
		return out