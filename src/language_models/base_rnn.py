"""Base RNN Cells
"""
import torch.nn as nn
import src.constants as C

class BaseRNN(nn.Module):
    def __init__(self, vocab_size, h_size, max_input_len, rnn_cell, n_layers):
        super(BaseRNN, self).__init__()
        self.n_size = vocab_size
        self.max_input_len = max_input_len
        self.h_size = h_size
        self.rnn_cell = {C.RNN_CELL_LSTM: nn.LSTM, C.RNN_CELL_GRU: nn.GRU}[rnn_cell]

    def forward(self, *args, **kwargs):
        raise NotImplementedError()
