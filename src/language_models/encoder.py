"""Encoder for LM
"""

import torch.nn as nn

from src.language_models.base_rnn import BaseRNN
import src.constants as C

class Encoder(BaseRNN):
    """Encoder for questions and reviews
    """

    def __init__(self, vocab_size, h_size, max_len, rnn_cell=C.RNN_CELL_LSTM, n_layers=1):
        super(Encoder, self).__init__(vocab_size, h_size, max_len, rnn_cell, n_layers)

        self.embedding = nn.Embedding(vocab_size, h_size)
        self.rnn = self.rnn_cell(h_size, h_size, n_layers, batch_first=True)

    def forward(self, input_seqs, input_lengths):
        embedded = self.embedding(input_seqs)
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden
