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
from models.seq2seq import Seq2Seq
from trainer.loss import Loss

from evaluator.evaluator import COCOEvalCap


USE_CUDA = torch.cuda.is_available()

class TrainerMetrics:

    def __init__(self, logger):
        self.train_loss = []
        self.dev_loss = []

        self.train_perplexity = []
        self.dev_perplexity = []

        self.logger = logger

    def add_loss(self, loss, mode):
        epoch_loss = loss.epoch_loss()
        epoch_perplexity = loss.epoch_perplexity()

        if mode == C.TRAIN_TYPE:
            self.train_loss.append(epoch_loss)
            self.train_perplexity.append(epoch_perplexity)
            min_loss, min_perplexity = np.nanmin(self.train_loss), np.nanmin(self.train_perplexity)
        elif mode == C.DEV_TYPE:
            self.dev_loss.append(epoch_loss)
            self.dev_perplexity.append(epoch_perplexity)
            min_loss, min_perplexity = np.nanmin(self.dev_loss), np.nanmin(self.dev_perplexity)
        else:
            raise 'Unimplemented mode: %s' % mode

        mode = mode.upper()
        self.logger.log('\n\t[%s] Loss = %.4f, Min [%s] Loss = %.4f' % (mode, epoch_loss, mode, min_loss))
        self.logger.log('\t[%s] Perplexity = %.2f, Min [%s] Perplexity = %.2f' % (mode, epoch_perplexity, mode, min_perplexity))

    def is_best_dev_loss(self):
        return float(len(self.dev_loss)) > 0.0 and float(self.dev_loss[-1]) == float(np.nanmin(self.dev_loss))

class Trainer:

    def __init__(self, 
        dataloader, params,
        save_model_every=1,     # Every Number of epochs to save after
        print_every=1000,       # Every Number of batches to print after
        dev_loader=None,
        #test_loader=None,
        vocab=None,
        saver=None,
        resume_training=False,
        resume_epoch=None
    ):

        self.save_model_every = save_model_every
        self.print_every = print_every
        self.params = params
        self.vocab = vocab
        self.model_name = params[C.MODEL_NAME]
        self.start_epoch = 0
        self.resume_training = resume_training
        self.lr = None

        # Data Loaders
        self.dataloader = dataloader
        self.dev_loader = dev_loader
        #self.test_loader = test_loader

        # Saver and Logger
        self.saver = saver
        self.logger = self.saver.logger

        # Model
        self.model = Seq2Seq(
            self.vocab.get_vocab_size(),
            hsizes(params, self.model_name),
            params,
        ) if self.dataloader else None

        self.logger.log('MODEL : %s' % self.model)
        self.logger.log('PARAMS: %s' % self.params)

        # Optimizer and loss metrics
        if self.resume_training:
            self.optimizer, self.metrics = self.saver.load_model_and_state(resume_epoch, self.model)
            self.start_epoch = resume_epoch + 1
        else:
            self.optimizer = None
            self.metrics = TrainerMetrics(self.logger)

        self.loss = Loss()
        if USE_CUDA:
            if self.model:
                self.model = self.model.cuda()


    def train(self):
        self.lr = self.params[C.LR]
        self._set_optimizer(0)

        # Save params, vocab and architecture
        self.saver.save_params_and_vocab(self.params, self.vocab, str(self.model))

        # For Debuging
        # self.dataloader = list(self.dataloader)[:10]
        # self.dev_loader = list(self.dev_loader)[:5]
        # self.test_loader = list(self.test_loader)[:5]

        if not self.resume_training:
            self.logger.log('Evaluating on DEV before epoch : 0')
            self.eval(self.dev_loader, C.DEV_TYPE, epoch=-1)

            # Add train loss entry for a corresponding dev loss entry before epoch 0
            self.loss.reset()
            #self.metrics.add_loss(self.loss, C.TRAIN_TYPE)
            self.eval(self.dataloader, C.TRAIN_TYPE, epoch=-1)

        for epoch in range(self.start_epoch, self.params[C.NUM_EPOCHS]):
            self.logger.log('\n  --- STARTING EPOCH : %d --- \n' % epoch)

            # refresh loss, perplexity 
            self.loss.reset()
            for batch_itr, inputs in enumerate(tqdm(self.dataloader)):
                answer_seqs, question_seqs, question_ids, review_seqs, \
                    answer_lengths = _extract_input_attributes(inputs, self.model_name)
                batch_loss, batch_perplexity = self.train_batch(
                    question_seqs,
                    review_seqs,
                    answer_seqs,
                    answer_lengths
                )
                if batch_itr % self.print_every == 0:
                    self.logger.log('\n\tMean [TRAIN] Loss for batch %d = %.2f' % (batch_itr, batch_loss))
                    self.logger.log('\tMean [TRAIN] Perplexity for batch %d = %.2f' % (batch_itr, batch_perplexity))

            self.logger.log('\n  --- END OF EPOCH : %d --- \n' % epoch)
            # Compute epoch loss and perplexity
            self.metrics.add_loss(self.loss, C.TRAIN_TYPE)

            # Eval on dev set
            self.logger.log('\nStarting evaluation on DEV at end of epoch: %d' % epoch)
            self.eval(self.dev_loader, C.DEV_TYPE, epoch=epoch)
            self.logger.log('Finished evaluation on DEV')

            # Save model periodically
            if epoch % self.save_model_every == 0:
                self.saver.save_model_and_state(epoch,
                    self.model,
                    self.optimizer,
                    self.metrics
                )
            # Update lr is the val loss increases
            if self.params[C.LR_DECAY] is not None and epoch >= self.params[C.DECAY_START_EPOCH] - 1:
                if epoch > 0 and self.metrics.dev_loss[-1] > self.metrics.dev_loss[epoch-1]:
                    self._decay_lr(epoch, self.params[C.LR_DECAY])
                    lr *= self.params[C.LR_DECAY]
                    self._set_optimizer(epoch, lr=lr)

            # Save the best model till now
            if self.metrics.is_best_dev_loss():
                self.saver.save_model(C.BEST_EPOCH_IDX, self.model)


    def train_batch(self, 
            question_seqs,
            review_seqs,
            answer_seqs,
            answer_lengths
        ):
        # Set model in train mode
        self.model.train(True)

        # Zero grad and teacher forcing
        self.optimizer.zero_grad()
        teacher_forcing_ratio = self.params[C.TEACHER_FORCING_RATIO]

        # run forward pass
        loss, perplexity, _, _ = self._forward_pass(question_seqs, review_seqs, answer_seqs, teacher_forcing_ratio)

        # gradient computation
        loss.backward()

        # update parameters
        params = itertools.chain.from_iterable([g['params'] for g in self.optimizer.param_groups])
        torch.nn.utils.clip_grad_norm(params, self.params[C.GLOBAL_NORM_MAX])
        self.optimizer.step()

        return loss.data.item(), perplexity


    def eval(self, dataloader, mode, output_filename=None, epoch=0):

        self.model.eval()

        if not dataloader:
            raise 'No [%s] Dataset' % mode
        else:
            self.logger.log('Evaluating on [%s] dataset (epoch %d)' % (mode.upper(), epoch))

        compute_loss = mode != C.TEST_TYPE
        if compute_loss:
            self.loss.reset()

        gold_answers_dict = {}
        generated_answer_dict = {}

        for batch_itr, inputs in tqdm(enumerate(dataloader)):
            answer_seqs, question_seqs, question_ids, review_seqs, \
                answer_lengths = _extract_input_attributes(inputs, self.model_name)

            _, _, output_seq, output_lengths = self._forward_pass(
                question_seqs,
                review_seqs,
                answer_seqs,
                1.0,
                compute_loss=compute_loss
            )

            assert len(question_ids) == len(output_lengths)

            if mode == C.TEST_TYPE:
                output_seq = output_seq.data.cpu().numpy()
                with open(output_filename, 'a') as fp:
                    for seq_itr, length in enumerate(output_lengths):
                        length = int(length)
                        seq = output_seq[seq_itr, :length]
                        if seq[-1] == C.EOS_INDEX:
                            seq = seq[:-1]
                        tokens = self.vocab.token_list_from_indices(seq)
                        generated_answer = ' '.join(tokens)
                        fp.write(generated_answer + '\n')
                        
                        gold_answers = []
                        question_id = question_ids[seq_itr]
                        answer_ids = dataloader.questionAnswersDict[question_id]
                        for answer_id in answer_ids:
                            answer_seq = dataloader.answersDict[answer_id]
                            answer_tokens = self.vocab.token_list_from_indices(answer_seq)
                            gold_answers.append(' '.join(answer_tokens))

                        gold_answers_dict[question_id] = gold_answers
                        generated_answer_dict[question_id] = [generated_answer]
        
                print(COCOEvalCap.compute_scores(gold_answers_dict, generated_answer_dict))

        if mode == C.DEV_TYPE:
            self.metrics.add_loss(self.loss, C.DEV_TYPE)
        elif mode == C.TEST_TYPE:
            self.logger.log('Saving generated answers to file {0}'.format(output_filename))
        elif mode == C.TRAIN_TYPE:
            self.metrics.add_loss(self.loss, C.TRAIN_TYPE)
        else:
            raise 'Unimplemented mode: %s' % mode

    def _forward_pass(self,
            question_seqs,
            review_seqs,
            answer_seqs,
            teacher_forcing_ratio,
            compute_loss=True
        ):
        target_seqs, answer_seqs  = _var(answer_seqs), _var(answer_seqs)
        question_seqs = None if self.model_name == C.LM_ANSWERS else _var(question_seqs)
        review_seqs = map(_var, review_seqs) if self.model_name == C.LM_QUESTION_ANSWERS_REVIEWS else None

        # run forward pass
        outputs, output_hidden, ret_dict = self.model(
            question_seqs,
            review_seqs,
            answer_seqs,
            target_seqs,
            teacher_forcing_ratio
        )

        output_seq = ret_dict['sequence']
        output_lengths = ret_dict['length']
        output_seq = torch.cat(output_seq, 1)
        
        # loss and gradient computation
        loss, perplexity = None, None
        if compute_loss:
            loss, perplexity = self.loss.eval_batch_loss(outputs, target_seqs)

        return loss, perplexity, output_seq, output_lengths

    def _set_optimizer(self, epoch):
        opt_type = self.params[C.OPTIMIZER_TYPE]
        if opt_type == C.ADAM:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif opt_type == C.SGD:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise 'Unimplemented optimization type: %s' % opt_type

        self.logger.log('\nSetting [%s] Learning Rate = %.6f (Epoch = %d)' % (opt_type.upper(), self.lr, epoch))

    def _decay_lr(self, epoch, decay_factor):
        opt_type = self.params[C.OPTIMIZER_TYPE]
        self.logger.log('\nDecaying [%s] learning rate by %.3f (Epoch = %d)' % (opt_type.upper(), decay_factor, epoch))

        if opt_type == C.ADAM:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= decay_factor
        elif opt_type == C.SGD:
            self.lr *= decay_factor
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise 'Unimplemented optimization type: %s' % opt_type

def _var(variable):
    dtype = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
    return Variable(torch.LongTensor(variable).type(dtype))

def _extract_input_attributes(inputs, model_name):
    if model_name == C.LM_ANSWERS:
        answer_seqs, answer_lengths = inputs
        question_seqs, review_seqs, question_ids = None, None, None
    elif model_name == C.LM_QUESTION_ANSWERS:
        (answer_seqs, answer_lengths), question_seqs, question_ids = inputs
        review_seqs = None
    elif model_name == C.LM_QUESTION_ANSWERS_REVIEWS:
        (answer_seqs, answer_lengths), question_seqs, question_ids, review_seqs = inputs
    else:
        raise 'Unimplemented model: %s' % model_name

    return answer_seqs, question_seqs, question_ids, review_seqs, answer_lengths

def hsizes(params, model_name):
    r_hsize, q_hsize, a_hsize = params[C.HDIM_R], params[C.HDIM_Q], params[C.HDIM_A]
    if model_name == C.LM_QUESTION_ANSWERS:
        assert a_hsize == q_hsize
    if model_name == C.LM_QUESTION_ANSWERS_REVIEWS:
        # TODO Fix this workaround
        if params[C.USE_ATTENTION]:
            assert a_hsize == r_hsize == q_hsize
        else:
            assert a_hsize == r_hsize + q_hsize
    return r_hsize, q_hsize, a_hsize

