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

    def __init__(self, vocab_size, h_size, max_len, n_layers, dropout_p, model):
        super(LM, self).__init__()

        embedding = nn.Embedding(vocab_size, h_size)

        self.model = model
        self.question_encoder = None if model == C.LM_ANSWERS else Encoder(
            vocab_size, h_size,
            max_len, n_layers, 
            dropout_p, embedding=embedding
        )
        self.reviews_encoder = Encoder(
            vocab_size, h_size,
            max_len, n_layers, 
            dropout_p, embedding=embedding
        ) if model == C.LM_QUESTION_ANSWERS_REVIEWS else None

        self.decoder = Decoder(vocab_size, h_size, max_len, n_layers, dropout_p)

    def forward(self,
        question_seqs,
        review_seqs,
        answer_seqs,
        teacher_forcing
    ):
        if self.model == C.LM_ANSWERS:
            d_hidden = None
        elif self.model == C.LM_QUESTION_ANSWERS:
            _, d_hidden = self.encoder(answer_seqs)
        elif self.model == C.LM_QUESTION_ANSWERS_REVIEWS:
            _, question_hidden = self.encoder(question_seqs)
            reviews_hidden = [self.encoder(seq)[1] for seq in review_seqs]
            reviews_hidden = map(_mean, zip(*reviews_hidden))
        else:
            raise 'Unimplemented model: %s' % self.model

        return self.decoder(target_seqs, d_hidden, teacher_forcing)

def _mean(vars):
    return torch.mean(torch.cat([i.unsqueeze(0) for i in vars], 0), 0)