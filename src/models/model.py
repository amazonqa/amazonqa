"""Seq2seq model for language models
"""

import numpy as np
import torch
import torch.nn as nn

from models.base_rnn import BaseRNN
import constants as C

from models.encoder import Encoder
from models.decoder import Decoder

class LM(nn.Module):

    def __init__(self, vocab_size, h_size, e_size, max_len, n_layers, dropout_p, model):
        super(LM, self).__init__()

        r_hsize, q_hsize, a_hsize = h_size
        embedding = nn.Embedding(vocab_size, e_size)

        self.model = model
        self.question_encoder = None if model == C.LM_ANSWERS else Encoder(
            vocab_size, q_hsize, e_size,
            max_len, n_layers,
            dropout_p, embedding=embedding
        )
        self.reviews_encoder = Encoder(
            vocab_size, r_hsize, e_size,
            max_len, n_layers,
            dropout_p, embedding=embedding
        ) if model == C.LM_QUESTION_ANSWERS_REVIEWS else None

        if self.model == C.LM_QUESTION_ANSWERS:
            assert q_hsize == a_hsize
        if self.model == C.LM_QUESTION_ANSWERS_REVIEWS:
            assert a_hsize == q_hsize + r_hsize

        self.decoder = Decoder(
            vocab_size, a_hsize, e_size,
            max_len, n_layers,
            dropout_p, embedding=embedding
        )

    def forward(self,
        question_seqs,
        review_seqs,
        answer_seqs,
        target_seqs,
        teacher_forcing
    ):
        #print(question_seqs, review_seqs, answer_seqs, target_seqs)
        if self.model == C.LM_ANSWERS:
            d_hidden = None
        elif self.model == C.LM_QUESTION_ANSWERS:
            _, d_hidden = self.question_encoder(question_seqs)
        elif self.model == C.LM_QUESTION_ANSWERS_REVIEWS:
            _, question_hidden = self.question_encoder(question_seqs)
            reviews_hidden = [self.reviews_encoder(seq)[1] for seq in review_seqs]
            reviews_hidden = list(map(_mean, zip(*reviews_hidden)))
            d_hidden = tuple(torch.cat([q_h, r_h], 2) for q_h, r_h in zip(question_hidden, reviews_hidden))
        else:
            raise 'Unimplemented model: %s' % self.model

        return self.decoder(target_seqs, d_hidden, teacher_forcing)

def _mean(vars):
    return torch.mean(torch.cat([i.unsqueeze(0) for i in vars], 0), 0)

# def _cat_hidden(h1, h2):
#     return (torch.cat([h])