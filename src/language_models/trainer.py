"""Trainer module for training seq2seq model
"""

import json
import os
import pickle
from datetime import datetime
from tqdm import tqdm
import itertools

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
        print_every=1000,       # Every Number of batches to print after
        dev_loader=None,
        test_loader=None,
        vocab=None,
        logger=None
    ):
        _set_random_seeds(random_seed)

        self.save_model_every = save_model_every
        self.print_every = print_every
        self.params = params
        self.vocab = vocab
        self.model_name = params[C.MODEL_NAME]

        # Data Loaders
        self.dataloader = dataloader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        # Logger
        self.logger = logger

        # Model
        self.model = LM(
            self.vocab.get_vocab_size(),
            hsizes(params[C.HDIM], self.model_name),
            params[C.EMBEDDING_DIM],
            params[C.OUTPUT_MAX_LEN],
            params[C.H_LAYERS],
            params[C.DROPOUT],
            params[C.MODEL_NAME]
        ) if self.dataloader else None
        self.logger.log('MODEL : %s' % self.model)
        self.logger.log('PARAMS: %s' % self.params)

        # Saving params
        self.save_dir = self._save_dir(datetime.now())

        # Optimizer and loss metrics
        self.optimizer = None
        self.loss = None
        self.perplexity = None
        self.criterion = nn.NLLLoss(ignore_index=C.PAD_INDEX)

        self.optimizer = None

        if USE_CUDA:
            if self.model:
                self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()


    def train(self):
        lr = self.params[C.LR]
        self._set_optimizer(0, lr)
        self.save_metadata()

        # For Debuging
        # self.dataloader = list(self.dataloader)[:10]
        # self.dev_loader = list(self.dev_loader)[:5]
        # self.test_loader = list(self.test_loader)[:5]

        for epoch in range(self.params[C.EPOCHS]):
            # refresh loss, perplexity 
            self.loss, self.perplexity = [], []

            self.logger.log('Epoch: %d' % epoch)
            for batch_itr, inputs in enumerate(tqdm(self.dataloader)):
                answer_seqs, quesion_seqs, review_seqs, \
                    answer_lengths = _extract_input_attributes(inputs, self.model_name)
                loss = self.train_batch(
                    quesion_seqs,
                    review_seqs,
                    answer_seqs,
                    answer_lengths
                )
                self.loss.append(loss)
                self.perplexity.append(_perplexity_from_loss(loss))

                if batch_itr % self.print_every == 0:
                    self.logger.log('\n[Training] Loss at batch %d = %.2f' % (batch_itr, self.loss[-1]))
                    self.logger.log('[Training] Perplexity at batch %d = %.2f' % (batch_itr, self.perplexity[-1]))
                    _print_info(epoch, batch_itr, self.loss, self.perplexity, 'Training', self.logger)

            # Save model
            if epoch % self.save_model_every == 0:
                self.save_model(epoch)
            if epoch >= self.params[C.DECAY_START_EPOCH]:
                lr *= self.params[C.LR_DECAY]
                self._set_optimizer(epoch, lr=lr)
            _print_info(epoch, -1, self.loss, self.perplexity, 'Training (EPOCH COMPLETED)', self.logger)

            # Eval on dev and test sets
            self.logger.log('Evaluating on DEV and TEST at end of epoch: %d' % epoch)
            self.eval(self.dev_loader, C.DEV_TYPE)
            self.eval(
                self.test_loader,
                C.TEST_TYPE,
                output_filename=self._output_filename(epoch)
            )

    def train_batch(self, 
            quesion_seqs,
            review_seqs,
            answer_seqs,
            answer_lengths
        ):

        # Set model in train mode
        self.model.train(True)

        # Zero grad and teacher forcing
        self.optimizer.zero_grad()
        teacher_forcing = np.random.random() < self.params[C.TEACHER_FORCING_RATIO]

        # run forward pass
        loss, _, _ = self._forward_pass(
            quesion_seqs,
            review_seqs,
            answer_seqs,
            teacher_forcing
        )

        # gradient computation
        loss.backward()

        # update parameters
        params = itertools.chain.from_iterable([g['params'] for g in self.optimizer.param_groups])
        torch.nn.utils.clip_grad_norm(params, self.params[C.GLOBAL_NORM_MAX])
        self.optimizer.step()

        return loss.data[0]

    def eval(self, dataloader, mode, output_filename=None):

        self.model.eval()
        losses, perplexities = [], []

        if not dataloader:
            raise 'No [%s] Dataset' % mode
        else:
            self.logger.log('Evaluating on [%s] dataset' % mode)

        compute_loss = mode != C.TEST_TYPE

        for batch_itr, inputs in tqdm(enumerate(dataloader)):
            answer_seqs, quesion_seqs, review_seqs, \
                answer_lengths = _extract_input_attributes(inputs, self.model_name)

            loss, output_seq, output_lengths = self._forward_pass(
                quesion_seqs,
                review_seqs,
                answer_seqs,
                False,
                compute_loss=compute_loss
            )

            if mode == C.DEV_TYPE:
                losses.append(loss.data[0])
                perplexities.append(_perplexity_from_loss(loss.data[0]))

                if batch_itr % self.print_every == 0:
                    self.logger.log('\n[%s] Loss at batch %d = %.2f' % (mode, batch_itr, losses[-1]))
                    self.logger.log('[%s] Perplexity at batch %d = %.2f' % (mode, batch_itr, perplexities[-1]))

            if mode == C.TEST_TYPE:
                output_seq = output_seq.data.cpu().numpy()
                with open(output_filename, 'a') as fp:
                    for seq_itr, length in enumerate(output_lengths):
                        length = int(length)
                        seq = output_seq[seq_itr, :length]
                        if seq[-1] == C.EOS_INDEX:
                            seq = seq[:-1]
                        tokens = self.vocab.token_list_from_indices(seq)
                        fp.write(' '.join(tokens) + '\n')

        if mode == C.DEV_TYPE:
            _print_info(1, -1, losses, perplexities, mode, self.logger)
        elif mode == C.TEST_TYPE:
            self.logger.log('Saving generated answers to file {0}'.format(output_filename))
        else:
            raise 'Unimplemented mode: %s' % mode
        return

    def _forward_pass(self,
            quesion_seqs,
            review_seqs,
            answer_seqs,
            teacher_forcing,
            compute_loss=True
        ):
        target_seqs, answer_seqs  = _var(answer_seqs), _var(answer_seqs)
        quesion_seqs = None if self.model_name == C.LM_ANSWERS else _var(quesion_seqs)
        review_seqs = map(_var, review_seqs) if self.model_name == C.LM_QUESTION_ANSWERS_REVIEWS else None

        # run forward pass
        outputs, output_seq, output_lengths = self.model(
            quesion_seqs,
            review_seqs,
            answer_seqs,
            target_seqs,
            teacher_forcing
        )

        # loss and gradient computation
        loss = _batch_loss(self.criterion, outputs, target_seqs)  if compute_loss else None

        return loss, output_seq, output_lengths

    def save_model(self, epoch):
        model_filename = '%s/%s_%d' % (self.save_dir, C.SAVED_MODEL_FILENAME, epoch)
        self.logger.log('Saving model (Epochs = %s)...' % epoch)
        torch.save(self.model.state_dict(), model_filename)

    def _output_filename(self, epoch):
        _ensure_path(self.save_dir)
        return '%s/generated_answers_%d.txt' % (self.save_dir, epoch)

    def save_metadata(self):
        _ensure_path(self.save_dir)
        params_filename = '%s/%s' % (self.save_dir, C.SAVED_PARAMS_FILENAME)
        vocab_filename = '%s/%s' % (self.save_dir, C.SAVED_VOCAB_FILENAME)
        architecture_filename = '%s/%s' % (self.save_dir, C.SAVED_ARCHITECTURE_FILENAME)

        self.logger.log('Saving params in file: %s' % params_filename)
        with open(params_filename, 'w') as fp:
            json.dump(self.params, fp, indent=4, sort_keys=True)

        self.logger.log('Saving vocab in file: %s' % vocab_filename)
        with open(vocab_filename, 'wb') as fp:
            pickle.dump(self.vocab, fp, pickle.HIGHEST_PROTOCOL)

        self.logger.log('Saving architecture in file: %s' % architecture_filename)
        with open(architecture_filename, 'w') as fp:
            fp.write(str(self.model))

    def _save_dir(self, time):
        time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
        return '%s/%s/%s/%s' % (C.BASE_PATH, self.params[C.CATEGORY], self.params[C.MODEL_NAME], time_str)

    def _set_optimizer(self, epoch, lr):
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.logger.log('Setting Learning Rate = %.3f (Epoch = %d)' % (lr, epoch))

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
    return np.exp(loss)

def _print_info(epoch, batch, losses, perplexities, corpus, logger):
    logger.log('Epoch = %d, Batch = %d, [%s] Loss = %.2f' % (epoch, batch, corpus, np.mean(np.array(losses))))
    logger.log('Epoch = %d, Batch = %d, [%s] Perplexity = %.2f' % (epoch, batch, corpus, np.mean(np.array(perplexities))))
    logger.log('\n')

def _var(variable):
    dtype = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
    return Variable(torch.LongTensor(variable).type(dtype))

def _extract_input_attributes(inputs, model_name):
    if model_name == C.LM_ANSWERS:
        answer_seqs, answer_lengths = inputs
        quesion_seqs, review_seqs = None, None
    elif model_name == C.LM_QUESTION_ANSWERS:
        (answer_seqs, answer_lengths), quesion_seqs = inputs
        review_seqs = None
    elif model_name == C.LM_QUESTION_ANSWERS_REVIEWS:
        (answer_seqs, answer_lengths), quesion_seqs, review_seqs = inputs
    else:
        raise 'Unimplemented model: %s' % model_name

    return answer_seqs, quesion_seqs, review_seqs, answer_lengths

def hsizes(hdim, model_name):
    if model_name == C.LM_ANSWERS:
        sizes = (None, None, hdim)
    elif model_name == C.LM_QUESTION_ANSWERS:
        sizes = (None, hdim, hdim)
    elif model_name == C.LM_QUESTION_ANSWERS_REVIEWS:
        sizes = (hdim, hdim, 2 * hdim)
    else:
        raise 'Unimplemented model: %s' % self.model_name
    return sizes
