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
        batch_num_tokens = (targets.cpu().data.numpy() != C.PAD_INDEX).sum()
        assert batch_size > 0

        # Add to num sentences and tokens since reset
        self.total_num_sentences += batch_num_sentences
        self.total_num_tokens += batch_num_tokens
        self.total_num_batches += 1

        batch_loss = self.criterion(outputs.continous().view(-1, outputs.size(2)), targets.continous().view(-1))
        batch_loss = batch_loss.data[0].item()

        self.total_loss += batch_loss

        if loss_type == C.WORD_LOSS:
            loss = batch_loss / batch_num_tokens
        elif loss_type == C.SENTENCE_LOSS:
            loss = batch_loss / batch_num_sentences
        else:
            raise 'Unimplemented Loss Type: %s' % loss_type
         
        return loss, _perplexity(batch_loss, batch_num_tokens)


    def epoch_loss(self):
        """NLL loss per sentence since the last reset
        """
        return self.total_loss / self.total_num_sentences if self.total_num_sentences > 0 else np.nan


    def epoch_perplexity(self):
        """Corpus perplexity per token since the last reset
        """
        return _perplexity(self.total_loss, self.total_num_tokens)


def _perplexity(loss, num_tokens):
    return np.exp(loss / total_num_tokens) if total_num_tokens > 0 else np.nan


