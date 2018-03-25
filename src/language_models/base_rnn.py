"""Base RNN Cells
"""
import torch.nn as nn
import src.constants as C

class BaseRNN(nn.Module):
    def __init__(self, vocab_size, h_size, max_len, rnn_cell, n_layers):
        super(BaseRNN, self).__init__()
        self.n_size = vocab_size
        self.max_len = max_len
        self.h_size = h_size
        self.rnn_cell = {C.RNN_CELL_LSTM: nn.LSTM, C.RNN_CELL_GRU: nn.GRU}[rnn_cell]
        self.n_layers = n_layers

    def forward(self, *args, **kwargs):
        raise NotImplementedError()
