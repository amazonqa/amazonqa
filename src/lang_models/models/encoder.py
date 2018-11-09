"""Encoder for LM
"""

import torch.nn as nn

from models.base_rnn import BaseRNN
import constants as C

class Encoder(BaseRNN):
    """Encoder for questions and reviews
    """

    def __init__(self, vocab_size, h_size, e_size,
        max_len, n_layers, dropout_p,
        embedding=None,
        rnn_cell=C.RNN_CELL_LSTM
    ):
        super(Encoder, self).__init__(vocab_size, h_size, e_size,
            max_len, rnn_cell, n_layers,
            dropout_p
        )

        self.embedding = embedding if embedding else nn.Embedding(vocab_size, e_size)
        self.rnn = self.rnn_cell(e_size, h_size, n_layers, dropout=self.dropout_p, batch_first=True)

    def forward(self, input_seqs):
        embedded = self.embedding(input_seqs)
        return self.rnn(embedded)
