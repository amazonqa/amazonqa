import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.optim import Adam

import data
import model
import constants as C

is_cuda = True if torch.cuda.is_available() else False

answerCorpus = data.AnswersCorpus(C.VIDEO_GAMES)
vocab_size = len(answerCorpus.dictionary)

batch_size = 40
train_loader = data.ShuffleDataLoader(answerCorpus.train, batch_size)
val_loader = data.ShuffleDataLoader(answerCorpus.val, batch_size)
test_loader = data.ShuffleDataLoader(answerCorpus.test, batch_size)

learning_rate = 0.01
embedding_size = 400
hidden_size = 100
num_layers = 1

model = models.RNNModel(vocab_size, embedding_size, hidden_size, num_layers)

if is_cuda:
	model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

num_epochs = 1

for epoch in range(num_epochs):
	model.train()
	train_loss = 0

	for batch_idx, data in train_loader:
		inputs, labels = data

		inputs = Variable(inputs)
		labels = Variable(labels.view(-1))

		optimizer.zero_grad()

		outputs = model(inputs)
		loss = criterion(outputs.view(-1, vocab_size), labels)

		loss.backward()
		nn.utils.clip_grad_norm(model.parameters(), 0.25)

		optimizer.step()
		train_loss += loss.data[0]

		if batch_idx % 100 == 0:
			print(batch_idx, train_loss/(batch_idx+1))

	print("Epoch ", epoch, " Train Loss ", train_loss/length)

	model.eval()
	val_loss = 0
	length = len(val_loader)

	for batch_idx, data in enumerate(val_loader):
		inputs, labels = data
		inputs = Variable(inputs, volatile=False)
		labels = Variable(labels.view(-1), volatile=False)

		outputs = model(inputs)
		loss = criterion(outputs.view(-1, len(vocab)), labels)

		val_loss += loss.data[0]

	print("Epoch ", epoch, " Val Loss ", val_loss/length)
	torch.save(model.state_dict(), str(epoch)+'-model.pkl')
