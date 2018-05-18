"""Seq2seq model for language models
"""

import numpy as np
import torch
import torch.nn as nn

import constants as C

from models.DecoderRNN import DecoderRNN
from models.EncoderRNN import EncoderRNN
from models.baseRNN import BaseRNN


class Seq2Seq(nn.Module):

    def __init__(self, vocab_size, h_sizes, params):
        super(Seq2Seq, self).__init__()

        e_size, max_len, n_layers, dropout_p, model_name, use_attention = [params[i] for i in [C.EMBEDDING_DIM, C.OUTPUT_MAX_LEN, C.H_LAYERS, C.DROPOUT, C.MODEL_NAME, C.USE_ATTENTION]]
        r_hsize, q_hsize, a_hsize = h_sizes

        self.use_attention = use_attention
        self.model_name = model_name
        self.decoder = DecoderRNN(vocab_size=vocab_size, max_len=max_len, embedding_size=e_size, hidden_size=a_hsize,
                            n_layers=n_layers, dropout_p=dropout_p,
                            sos_id=C.SOS_INDEX, eos_id=C.EOS_INDEX, model_name=model_name, use_attention=self.use_attention)

        if model_name == C.LM_ANSWERS:
            self.question_encoder = None
        else:
            self.question_encoder = EncoderRNN(vocab_size=vocab_size, max_len=max_len, embedding_size=e_size,
                        hidden_size=q_hsize, n_layers=n_layers, dropout_p=dropout_p)
            self.decoder.embedding.weight = self.question_encoder.embedding.weight

        if model_name == C.LM_QUESTION_ANSWERS_REVIEWS:
            self.reviews_encoder = EncoderRNN(vocab_size=vocab_size, max_len=max_len, embedding_size=e_size,
                        hidden_size=r_hsize, n_layers=n_layers, dropout_p=dropout_p)
            self.decoder.embedding.weight = self.reviews_encoder.embedding.weight
        else:
            self.reviews_encoder = None


        if self.model_name == C.LM_QUESTION_ANSWERS:
            assert q_hsize == a_hsize
        if self.model_name == C.LM_QUESTION_ANSWERS_REVIEWS:
            # TODO Fix this workaround
            if self.use_attention:
                assert a_hsize == q_hsize == r_hsize
            else:
                assert a_hsize == q_hsize + r_hsize


    def forward(self,
        question_seqs,
        review_seqs,
        answer_seqs,
        target_seqs,
        teacher_forcing_ratio
    ):
        #print(question_seqs, review_seqs, answer_seqs, target_seqs)
        if self.model_name == C.LM_ANSWERS:
            d_hidden = None
            question_out = None
            review_outs = None
        elif self.model_name == C.LM_QUESTION_ANSWERS:
            question_out, d_hidden = self.question_encoder(question_seqs)
            review_outs = None
        elif self.model_name == C.LM_QUESTION_ANSWERS_REVIEWS:
            question_out, question_hidden = self.question_encoder(question_seqs)
            reviews_encoder_outs = [self.reviews_encoder(seq) for seq in review_seqs]
            review_outs, review_hiddens = map(list, zip(*reviews_encoder_outs))

            # TODO Fix this workaround
            if self.use_attention:
                d_hidden = question_hidden
            else:
                reviews_hidden = list(map(_mean, zip(*review_hiddens)))
                d_hidden = tuple(torch.cat([q_h, r_h], 2) for q_h, r_h in zip(question_hidden, reviews_hidden))
        else:
            raise 'Unimplemented model: %s' % self.model_name
        
        if self.use_attention:
            d_out = (question_out, review_outs)
        else:
            d_out = None
        return self.decoder(inputs=target_seqs, encoder_hidden=d_hidden, 
            encoder_outputs=d_out, teacher_forcing_ratio=teacher_forcing_ratio)

def _mean(vars):
    return torch.mean(torch.cat([i.unsqueeze(0) for i in vars], 0), 0)

# def _cat_hidden(h1, h2):
#     return (torch.cat([h])
