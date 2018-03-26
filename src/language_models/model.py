"""Seq2seq model for language models
"""

import numpy as np
import torch
import torch.nn as nn

from language_models.base_rnn import BaseRNN
import constants as C

from language_models.encoder import Encoder
from language_models.decoder import Decoder

class LM(nn.Module):

    def __init__(self, vocab_size, h_size, max_len, n_layers, dropout_p):
        super(LM, self).__init__()

        self.encoder = Encoder(vocab_size, h_size, max_len, n_layers, dropout_p)
        self.decoder = Decoder(vocab_size, h_size, max_len, n_layers, dropout_p)

    def forward(self, input_seqs, target_seqs, input_lengths, teacher_forcing):
        """Generates an output sequence using an encoder-decoder model
        Inputs:
            target_seqs: 
                padded target sequences of shape: batch_size x max_input_seq_len
            output_seqs: batch_size x output_seq_len
                when teacher forcing is true, it must have a sequence length of
                at least one i.e. shape = size batch_size x 1
                input_seq = Variable([SOS_INDEX] * batch_size).squeeze(1)
            input_lengths: lengths of inputs, 1D array of shape 1 x max_input_seq_len
            teacher_forcing: boolean, whether or not to use teacher forcing
        Outputs:
            decoder_outputs: softmax outputs from the decoder
            output_seqs: output sequences: batch_size x max_output_seq_len
            output_lengths: lengths of outputs used when teacher forcing is false
                np array, shape: 1 x max_output_seq_len
        """

        _, e_hidden = self.encoder(input_seqs, input_lengths)
        return self.decoder(target_seqs, e_hidden, teacher_forcing)
