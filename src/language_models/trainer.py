"""Trainer module for training seq2seq model
"""

import json
import os
import pickle
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

import constants as C
from language_models.model import LM

USE_CUDA = torch.cuda.is_available()

class Trainer:

    def __init__(self, 
        dataloader, params,
        random_seed=1, 
        save_model_every=1,     # Every Number of epochs to save after
        print_every=100,        # Every Number of batches to print after
        dev_loader=None,
        vocab=None
    ):
        _set_random_seeds(random_seed)

        self.save_model_every = save_model_every
        self.print_every = print_every
        self.params = params
        self.dataloader = dataloader
        self.dev_loader = dev_loader
        self.vocab = vocab
        self.model = LM(
            self.vocab.get_vocab_size(),
            params[C.HDIM],
            params[C.OUTPUT_MAX_LEN],
            params[C.H_LAYERS],
            params[C.DROPOUT],
            params[C.MODEL_NAME]
        )

        if USE_CUDA:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        self.loss = []
        self.perplexity = []
        self.criterion = nn.NLLLoss(ignore_index=C.PAD_INDEX)

        self.model_name = params[C.MODEL_NAME]
        self.optimizer = None


    def train_batch(self, 
            quesion_seqs,
            review_seqs,
            answer_seqs,
            answer_lengths
        ):
        self.optimizer.zero_grad()

        answer_seqs, target_seqs = _var(answer_seqs), _var(answer_seqs)

        quesion_seqs = None if self.model_name == C.LM_ANSWERS else _var(quesion_seqs)
        review_seqs = map(_var, review_seqs) if self.model_name == C.LM_QUESTION_ANSWERS_REVIEWS else None

        # run forward pass
        teacher_forcing = np.random.random() < self.params[C.TEACHER_FORCING_RATIO]
        outputs, _, _ = self.model(
            quesion_seqs,
            review_seqs,
            answer_seqs,
            target_seqs,
            teacher_forcing
        )
        #print(target_seqs.size())
        # loss and gradient computation
        loss = _batch_loss(self.criterion, outputs, target_seqs)
        loss.backward()

        # update parameters
        self.optimizer.step()

        return loss.data[0]

    def train(self):
        self._set_optimizer(0)

        for epoch in range(self.params[C.EPOCHS]):
            print('Epoch: %d', epoch)
            for batch_itr, inputs in enumerate(tqdm(self.dataloader)):
                if self.model_name == C.LM_ANSWERS:
                    answer_seqs, answer_lengths = inputs
                    quesion_seqs, review_seqs = None, None
                elif self.model_name == C.LM_QUESTION_ANSWERS:
                    (answer_seqs, answer_lengths), quesion_seqs = inputs
                    review_seqs = None
                elif self.model_name == C.LM_QUESTION_ANSWERS:
                    (answer_seqs, answer_lengths), quesion_seqs, review_seqs = inputs
                else:
                    raise 'Unimplemented model: %s' % self.model_name
                loss = self.train_batch(
                    quesion_seqs,
                    review_seqs,
                    answer_seqs,
                    answer_lengths
                )
                self.loss.append(loss)
                self.perplexity.append(_perplexity_from_loss(loss))
                if batch_itr % self.print_every == 0:
                    print('Loss at batch %d = %.2f' % (batch_itr, self.loss[-1]))
                    print('Perplexity at batch %d = %.2f' % (batch_itr, self.perplexity[-1]))
            if epoch % self.save_model_every == 0:
                self.save_model()
            if epoch == self.params[C.DECAY_START_EPOCH]:
                self._set_optimizer(epoch, lr_decay=self.params[C.LR_DECAY])

    def eval(self):
        dev_losses, dev_perplexities = [], []
        for batch_itr, inputs in tqdm(enumerate(self.dataloader)):
            if self.model_name == C.LM_ANSWERS:
                answer_seqs, _ = inputs
            elif self.model_name == C.LM_QUESTION_ANSWERS:
                (answer_seqs, _), quesion_seqs = inputs
            elif self.model_name == C.LM_QUESTION_ANSWERS:
                (answer_seqs, _), quesion_seqs, review_seqs = inputs
            else:
                raise 'Unimplemented model: %s' % self.model_name

            answer_seqs = _var(answer_seqs)
            quesion_seqs = None if self.model_name == C.LM_ANSWERS else _var(quesion_seqs)
            review_seqs = map(_var, review_seqs) if self.model_name == C.LM_QUESTION_ANSWERS_REVIEWS else None
            target_seqs = _var(answer_seqs)
            outputs, _, _ = self.model(
                quesion_seqs,
                review_seqs,
                answer_seqs,
                target_seqs,
                False
            )

            dev_loss = _batch_loss(self.criterion, outputs, target_seqs)
            dev_losses.append(dev_loss.data[0])
            dev_perplexities.append(_perplexity_from_loss(dev_loss.data[0]))
            if batch_itr % self.print_every == 0:
                print('[Dev] Loss at batch %d = %.2f', (batch_itr, dev_loss[-1]))
                print('[Dev] Perplexity at batch %d = %.2f', (batch_itr, dev_perplexities[-1]))
        _print_info(1, dev_losses, dev_perplexities, 'Development')
        return

    def save_model(self):
        save_dir = self._save_dir(datetime.now())
        _ensure_path(save_dir)

        model_filename = '%s/%s' % (save_dir, C.SAVED_MODEL_FILENAME)
        params_filename = '%s/%s' % (save_dir, C.SAVED_PARAMS_FILENAME)
        vocab_filename = '%s/%s' % (save_dir, C.SAVED_VOCAB_FILENAME)

        torch.save(self.model.state_dict(), model_filename)
        with open(params_filename, 'w') as fp:
            json.dump(self.params, fp, indent=4, sort_keys=True)
        
        with open(vocab_filename, 'wb') as fp:
            pickle.dump(self.vocab, fp, pickle.HIGHEST_PROTOCOL)

    def _save_dir(self, time):
        time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
        return '%s/%s/%s' % (C.BASE_PATH, self.params[C.MODEL_NAME], time_str)

    def _set_optimizer(self, epoch, lr_decay=1.0):
        lr = self.params[C.LR] * lr_decay
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        print('Setting Learning Rate = %.3f (Epoch = %d)' % (epoch, lr))

def _set_random_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def _batch_loss(criterion, outputs, targets):
    loss = 0
    # If the target is longer than max_output_len in
    # case of teacher_forcing = True,
    # then consider only max_output_len steps for loss
    n = min(len(outputs), targets.size(1) - 1)
    for idx in range(n):
        output = outputs[idx]
        loss += criterion(output, targets[:, idx + 1])
    return loss / n if n > 0 else 0

def _ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def _perplexity_from_loss(loss):
    return np.power(2.0, loss)

def _print_info(epoch, losses, perplexities, corpus):
    print('Epoch = %d, [%s] Loss = %.2f', (epoch, corpus, np.mean(np.array(losses))))
    print('Epoch = %d, [%s] Perplexity = %.2f', (epoch, corpus, np.mean(np.array(perplexities))))

def _var(variable):
    dtype = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
    return Variable(torch.LongTensor(variable).type(dtype))
