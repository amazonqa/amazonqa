"""Class for loss computation
"""

import constants as C
import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

class Loss:

    def __init__(self):
        self.criterion = nn.NLLLoss(ignore_index=C.PAD_INDEX, size_average=False)

    def reset(self):
        self.num_tokens = 0
        self.num_sequences = 0
        self.total_loss = 0

    def batch_loss(self, outputs, targets):
        batch_size = targets.size(0)
        if batch_size == 0:
            return 0

        # sequences
        self.num_sequences += batch_size
        self.num_tokens += (targets.cpu().data.numpy() != C.PAD_INDEX).sum()
        loss = Variable(torch.zeros(1))

        # If the target is longer than max_output_len in
        # case of teacher_forcing = True,
        # then consider only max_output_len steps for loss
        n = min(len(outputs), targets.size(1) - 1)
        for idx in range(n):
            output = outputs[idx]
            loss += self.criterion(output, targets[:, idx + 1])

        self.total_loss += loss.data[0]
        return loss / batch_size

    def epoch_loss(self):
        """NLL loss per sequence since the last reset
        """
        return self.total_loss / self.num_sequences if self.num_sequences > 0 else 0
    
    def epoch_perplexity(self):
        """Corpus perplexity per token since the last reset
        """
        return np.exp(self.total_loss / self.num_tokens) if self.num_tokens > 0 else 0
