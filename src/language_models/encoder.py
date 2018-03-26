"""Encoder for LM
"""

import torch.nn as nn

from language_models.base_rnn import BaseRNN
import constants as C

class Encoder(BaseRNN):
    """Encoder for questions and reviews
    """

    def __init__(self, vocab_size, h_size, max_len, dropout_p, n_layers, embedding=None, rnn_cell=C.RNN_CELL_LSTM):
        super(Encoder, self).__init__(vocab_size, h_size, max_len, rnn_cell, n_layers, dropout_p)

        self.embedding = embedding if embedding else nn.Embedding(vocab_size, h_size)
        self.rnn = self.rnn_cell(h_size, h_size, n_layers, batch_first=True)

    def forward(self, input_seqs):
        embedded = self.embedding(input_seqs)
        embedded = self.dropout(embedded)
        return self.rnn(embedded)
