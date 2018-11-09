"""Base RNN Cells
"""
import torch.nn as nn
import constants as C

class BaseRNN(nn.Module):
    def __init__(self, vocab_size, h_size, e_size, max_len, rnn_cell, n_layers, dropout_p):
        super(BaseRNN, self).__init__()
        self.n_size = vocab_size
        self.max_len = max_len
        self.h_size = h_size
        self.e_size = e_size
        self.rnn_cell = {C.RNN_CELL_LSTM: nn.LSTM, C.RNN_CELL_GRU: nn.GRU}[rnn_cell]
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout_p)
        self.dropout_p = dropout_p

    def forward(self, *args, **kwargs):
        raise NotImplementedError()
