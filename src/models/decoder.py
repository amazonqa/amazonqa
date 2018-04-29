"""Decoder for LM
"""

import numpy as np
import torch
import torch.nn as nn

from models.base_rnn import BaseRNN
import constants as C

class Decoder(BaseRNN):
    """Decoder for answers
    """

    def __init__(self, vocab_size, h_size, e_size,
        max_len, n_layers, dropout_p,
        embedding=None,
        rnn_cell=C.RNN_CELL_LSTM
    ):
        super(Decoder, self).__init__(vocab_size, h_size, e_size,
            max_len, rnn_cell, n_layers, 
            dropout_p
        )

        self.embedding = embedding if embedding else nn.Embedding(vocab_size, e_size)
        self.rnn = self.rnn_cell(e_size, h_size, n_layers, dropout=self.dropout_p, batch_first=True)
        self.out = nn.Linear(h_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.output_seq_lengths = None
        self.decoder_outputs = None
        self.output_seq = None

    def softmax_from_input(self, input, hidden):
        batch_size, output_size = input.size()
        embedded = self.embedding(input)

        # RNN output: batch_size * seq_size * h_size
        output, hidden = self.rnn(embedded, hidden)

        # Softmax: batch_size * seq_size * vocab_size
        softmax = self.log_softmax(
            self.out(output.contiguous().view(-1, self.h_size))
        ).view(batch_size, output_size, -1)
        return softmax, hidden

    def symbol_from_softmax(self, idx, softmax_idx):
        """Returns the decoded symbol from softmax
        Idx is the index in the sequence
        softmax_idx is the softmax for idx of shape: batch_size x vocab_size
        output is symbols of shape: batch_size x 1
        """
        symbols = softmax_idx.exp().multinomial(1)

        # Update decoded symbols to 
        self.decoder_outputs.append(softmax_idx)
        self.output_seq.append(symbols)

        # If the current index is less than the output seq length AND current symbol is EOS, 
        # then update the output seq len to current index
        is_eos = (idx < self.output_seq_lengths) & (symbols.data.cpu().squeeze().numpy() == C.EOS_INDEX)
        self.output_seq_lengths[is_eos] = len(self.output_seq)

        return symbols

    def forward(self, target_seqs, hidden_intial, teacher_forcing):
        batch_size = target_seqs.size(0)
        hidden = hidden_intial

        self.output_seq_lengths = np.ones(batch_size) * self.max_len
        self.decoder_outputs = []
        self.output_seq = []

        if teacher_forcing:
            # input is a sequence, excluding the last token
            # i.e input <=> batch_size * (seq_len - 1) * h_size
            # output & hidden are of the same shape
            output, hidden = self.softmax_from_input(target_seqs[:, :-1], hidden)

            #print(output.size())
            for idx in range(output.size(1)):
                self.symbol_from_softmax(idx, output[:, idx, :])
        else:
            input = target_seqs[:, 0].unsqueeze(1)
            for idx in range(self.max_len):
                output, hidden = self.softmax_from_input(input, hidden)
                input = self.symbol_from_softmax(idx, output.squeeze(1))

        self.output_seq = torch.cat(self.output_seq, 1)
        return self.decoder_outputs, self.output_seq, self.output_seq_lengths
