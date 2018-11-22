"""Class for loss computation
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import constants as C

class Loss:

    def __init__(self):
        self.loss_type = C.WORD_LOSS

        if self.loss_type not in {C.WORD_LOSS, C.SENTENCE_LOSS}:
            raise 'Unimplemented Loss Type: %s' % self.loss_type

        self.criterion = nn.NLLLoss(ignore_index=C.PAD_INDEX, size_average=False)
        if C.USE_CUDA:
            self.criterion.cuda()


    def reset(self):
        self.total_num_tokens = 0
        self.total_num_sentences = 0
        self.total_num_batches = 0
        self.total_loss = 0


    def eval_batch_loss(self, outputs, targets):
        batch_num_sentences = targets.size(0)
        batch_num_tokens = targets.data.ne(C.PAD_INDEX).sum()
        #assert batch_size > 0

        # Add to num sentences and tokens since reset
        self.total_num_sentences += batch_num_sentences
        self.total_num_tokens += batch_num_tokens
        self.total_num_batches += 1

        outputs = torch.stack(outputs).transpose(0,1)
        batch_loss = self.criterion(outputs.contiguous().view(-1, outputs.size(2)), targets[:,1:].contiguous().view(-1))

        self.total_loss += batch_loss.data.item()

        if self.loss_type == C.WORD_LOSS:
            loss = batch_loss / float(batch_num_tokens)
        elif self.loss_type == C.SENTENCE_LOSS:
            loss = batch_loss / float(batch_num_sentences)

        return loss, _perplexity(batch_loss.data.item(), batch_num_tokens)

    def epoch_loss(self):
        """NLL loss per sentence since the last reset
        """
        if self.loss_type == C.WORD_LOSS:
            epoch_loss = self.total_loss / float(self.total_num_tokens)
        elif self.loss_type == C.SENTENCE_LOSS:
            epoch_loss = self.total_loss / float(self.total_num_sentences)

        return epoch_loss


    def epoch_perplexity(self):
        """Corpus perplexity per token since the last reset
        """
        return _perplexity(self.total_loss, self.total_num_tokens)


def _perplexity(loss, num_tokens):
    return np.exp(loss / float(num_tokens)) if num_tokens > 0 else np.nan

