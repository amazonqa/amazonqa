#from hyperparamters import *
from constants import *
import torch.nn as nn
from torch.nn import Sequential
import torch
from torch.autograd import Variable

batch_size = 64
#language model
embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 5
learning_rate = 0.001


dtype = torch.FloatTensor
ltype = torch.LongTensor
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor


class LM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, lstm_layers):
        super(LM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, lstm_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        return

    def forward(self, input):
        embed_out = self.embed(input)
        # Forward propagate RNN  
        out, (ht, ct) = self.lstm(x, h)
        print("OUT SHAPE - ", out.shape)
        print("HT SHAPE - ", ht.shape)
        print("CT SHAPE - ", ct.shape)
        # Reshape output to (batch_size*sequence_length, hidden_size)
        #out = out.contiguous().view(out.size(0)*out.size(1), out.size(2))
        # Decode hidden states of all time step
        out = self.linear(out)
        return out, h

