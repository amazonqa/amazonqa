"""Class for loss computation
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import constants as C

class Loss:

    def __init__(self):
        self.criterion = nn.NLLLoss(ignore_index=C.PAD_INDEX, size_average=False)
        if C.USE_CUDA:
            self.criterion.cuda()

    def reset(self):
        self.num_tokens = 0
        self.num_sequences = 0
        self.total_loss = 0

    def eval_batch_loss(self, outputs, targets):
        batch_size = targets.size(0)
        batch_num_tokens = (targets.cpu().data.numpy() != C.PAD_INDEX).sum()
        assert batch_size > 0

        # Add to num sequences and tokens since reset
        self.num_sequences += batch_size
        self.num_tokens += batch_num_tokens
        dtype = torch.cuda.FloatTensor if C.USE_CUDA else torch.FloatTensor
        loss = Variable(torch.zeros(1).dtype(dtype))

        # If the target is longer than max_output_len in
        # case of teacher_forcing = True,
        # then consider only max_output_len steps for loss
        n = min(len(outputs), targets.size(1) - 1)
        for idx in range(n):
            output = outputs[idx]
            loss += self.criterion(output, targets[:, idx + 1])

        self.total_loss += loss.data[0]
        return loss / batch_size, _perplexity(loss.data[0], batch_num_tokens)

    def epoch_loss(self):
        """NLL loss per sequence since the last reset
        """
        return self.total_loss / self.num_sequences if self.num_sequences > 0 else np.nan
    
    def epoch_perplexity(self):
        """Corpus perplexity per token since the last reset
        """
        return _perplexity(self.total_loss, self.num_tokens)

def _perplexity(loss, num_tokens):
    return np.exp(loss / num_tokens) if num_tokens > 0 else np.nan
