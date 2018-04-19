"""Save models, parameters and training states each optimization
"""

from datetime import datetime
import os
import pickle
import json
import torch
import pandas as pd

import constants as C
from logger import Logger

SAVED_PARAMS_FILENAME = 'params.json'
SAVED_VOCAB_FILENAME = 'vocab.pkl'
SAVED_ARCHITECTURE_FILENAME = 'architecture.txt'

SAVED_MODEL_FILENAME = 'model_%d.pt'
SAVED_STATE_FILENAME = 'trainer_state_%d.pt'
SAVED_METRICS_FILENAME = 'metrics_%d.tsv'

OPTIMIZER = 'optimizer'
METRICS = 'metrics'
USE_CUDA = torch.cuda.is_available()

class Saver:

    def __init__(self, save_dir, params):
        if save_dir:
            self.save_dir = save_dir
        else:
            self.save_dir = self._save_dir(datetime.now())
            self.params = params
    
        self.params_filename = '%s/%s' % (self.save_dir, SAVED_PARAMS_FILENAME)
        if save_dir:
            self.params = _json_load(self.params_filename)

        _ensure_path(self.save_dir)
        self.logger = Logger(clear_file=False, base_dir=self.save_dir)
        self.params[C.LOG_FILENAME] = self.logger.logfilename

        self.vocab_filename = '%s/%s' % (self.save_dir, SAVED_VOCAB_FILENAME)
        self.architecture_filename = '%s/%s' % (self.save_dir, SAVED_ARCHITECTURE_FILENAME)

    def save_params_and_vocab(self, params, vocab, architecture):
        self.logger.log('\nSaving params in file: %s' % self.params_filename)
        _json_dump(params, self.params_filename)

        self.logger.log('Saving vocab in file: %s' % self.vocab_filename)
        _pickle_dump(vocab, self.vocab_filename)

        self.logger.log('Saving architecture in file: %s' % self.architecture_filename)
        with open(self.architecture_filename, 'w') as fp:
            fp.write(architecture)

    def save_model(self, epoch, model):
        self.logger.log('\nSaving model (Epoch = %d)...' % epoch)
        model_filename = '%s/%s' % (self.save_dir, SAVED_MODEL_FILENAME % epoch)
        torch.save(model.state_dict(), model_filename)

    def load_model(self, epoch, model):
        map_location = None if USE_CUDA else lambda storage, loc: storage # assuming the model was saved from a gpu machine
        model_filename = '%s/%s' % (self.save_dir, SAVED_MODEL_FILENAME % epoch)
        self.logger.log('\nLoading model from epoch = %d...' % epoch)
        model.load_state_dict(torch.load(model_filename, map_location=map_location))

        if USE_CUDA:
            model.cuda()

    def save_model_and_state(self, epoch, model, optimizer, metrics):
        self.save_model(epoch, model)
        self.save_metrics(epoch, metrics)
        state_filename = '%s/%s' % (self.save_dir, SAVED_STATE_FILENAME % epoch)
        state = {
            OPTIMIZER: optimizer,
            METRICS: metrics
        }
        self.logger.log('Saving training state (Epoch = %d)...' % epoch)
        torch.save(state, state_filename)

    def load_model_and_state(self, epoch, model):
        self.load_model(epoch, model)
        state_filename = '%s/%s' % (self.save_dir, SAVED_STATE_FILENAME % epoch)
        self.logger.log('Loading training state from epoch %d...\n' % epoch)
        state = torch.load(state_filename)
        return state[OPTIMIZER], state[METRICS]

    def save_metrics(self, epoch, metrics):
        metrics_filename = '%s/%s' % (self.save_dir, SAVED_METRICS_FILENAME % epoch)
        pd.DataFrame({
            'Train Loss': metrics.train_loss,
            'Dev Loss': metrics.dev_loss,
            'Train Perplexity': metrics.train_perplexity,
            'Dev Perplexity': metrics.dev_perplexity
        }).to_csv(metrics_filename, sep='\t')

    def load_vocab(self):
        return _pickle_load(self.vocab_filename)

    def _save_dir(self, time):
        time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
        return '%s/%s/%s/%s' % (C.BASE_PATH, self.params[C.CATEGORY], self.params[C.MODEL_NAME], time_str)

def _ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def _pickle_dump(obj, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp, pickle.HIGHEST_PROTOCOL)

def _pickle_load(filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)

def _json_dump(obj, filename):
    with open(filename, 'w') as fp:
        json.dump(obj, fp, indent=4, sort_keys=True)

def _json_load(filename):
    with open(filename, 'r') as fp:
        return json.load(fp)
