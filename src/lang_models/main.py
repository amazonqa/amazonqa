"""Run different seq2seq models
"""

import os
import argparse
import pickle
import json
import torch
import numpy as np

import config
import constants as C
from trainer.trainer import Trainer, hsizes
from data.dataloader import AmazonDataLoader
from data.dataset import AmazonDataset
from models.seq2seq import Seq2Seq
from utils.logger import Logger
from utils.saver import Saver

RANDOM_SEED = 1
CACHE_DATASET = False

def main():
    _set_random_seeds(RANDOM_SEED)
    args = config.get_main_params()
    model_name, mode = args.model_name, args.mode
    save_dir = args.save_dir

    resume, epoch = args.resume, args.epoch

    if args.resume:
        assert mode == C.TRAIN_TYPE
        assert epoch >= 0

    params = config.get_model_params(model_name)
    params[C.MODEL_NAME] = model_name

    # Instantiate saver and a logger in save_dir
    # If save_dir is passed in from command line
    #   params are loaded from the save_dir
    # Logger is instantiated in saver
    saver = Saver(save_dir, params)
    logger = saver.logger
    params = saver.params

    # if save_dir is passed, 
    # model_name is used from the model_name in saved params
    model_name = params[C.MODEL_NAME]
    logger.log('SaveDir: %s' % saver.save_dir)

    if mode == C.TRAIN_TYPE:
        logger.log('\nLoading dataset..')
        dataset = AmazonDataset(params, mode)
        logger.log('\n Model: %s, Mode = %s \n' % (model_name, mode))
        logger.log('\nLoading dataloader..')

        if CACHE_DATASET:
            train_loader = pickle.load(open(model_name + 'train.pickle', 'rb'))
            dev_loader = pickle.load(open(model_name + 'dev.pickle', 'rb'))
        else:
            train_loader = AmazonDataLoader(dataset.train, model_name, params[C.BATCH_SIZE])
            dev_loader = AmazonDataLoader(dataset.val, model_name, params[C.BATCH_SIZE])
            pickle.dump(train_loader, open(model_name + 'train.pickle', 'wb'))
            pickle.dump(dev_loader, open(model_name + 'dev.pickle', 'wb'))

        logger.log('\nInstantiating training..')
        trainer = Trainer(
            train_loader,
            params,
            dev_loader=dev_loader,
            vocab=dataset.vocab,
            saver=saver,
            resume_training=resume,
            resume_epoch=epoch if resume else None
        )
        trainer.train()

    elif mode in [C.DEV_TYPE, C.TEST_TYPE]:
        logger.log('\nBeginning evaluation ..\n')

        # Load saved params and vocabs
        output_file = args.output_file
        logger.log('Loading vocab..')
        vocab = saver.load_vocab()
        model_name = params[C.MODEL_NAME]
        dataset = AmazonDataset(params, mode)
        #TODO: next line is a temporary change only. 
        dataset_typed = dataset.test
        #dataset_typed = dataset.val if mode == C.DEV_TYPE else dataset.test
        loader = AmazonDataLoader(dataset_typed, model_name, params[C.BATCH_SIZE])

        # Load model
        logger.log('Loading saved model..')
        model = Seq2Seq(
            vocab.get_vocab_size(),
            hsizes(params, model_name),
            params,
        )
        saver.load_model(epoch, model)

        # Instantiate trainer with saved model
        logger.log('Instantiating trainer..')
        trainer = Trainer(
            None,
                params,
            dev_loader=loader,
            saver=saver,
            vocab=vocab
        )
        logger.log('Adding model to trainer..')
        trainer.model = model

        # Evaluation on test set
        logger.log('Total number of [%s] batches: %d' % (mode.upper(), len(list(loader))))
        trainer.eval(loader, mode, output_filename=output_file)

        logger.log('\nCompleted Evaluation..\n')
    else:
        raise 'Unimplemented mode: %s' % mode

def _set_random_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == '__main__':
    main()
