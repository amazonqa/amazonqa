"""Run different seq2seq models
"""

import os
import argparse
import pickle
import json
import torch

import utils
from trainer import Trainer, hsizes
from dataloader import AmazonDataLoader
from dataset import AmazonDataset
from models.model import LM
from logger import Logger
import constants as C

RANDOM_SEED = 1

def main():

    args = utils.get_main_params()
    model_name, mode = args.model_name, args.mode
    save_dir = args.save_dir
    logger = Logger()

    resume, epoch = args.resume, args.epoch
    if args.resume:
        assert mode == C.TRAIN_TYPE
        assert epoch >= 0

    if mode == C.TRAIN_TYPE:
        params = utils.get_model_params(model_name)
        params[C.MODEL_NAME] = model_name
        params[C.LOG_FILENAME] = logger.logfilename
        category = params[C.CATEGORY]
        dataset = _get_dataset(model_name, category, params, logger)
        logger.log('\n Model: %s, Mode = %s, Category = %s \n' % (model_name, mode, category))
        train_loader = AmazonDataLoader(dataset.train, model_name, params[C.BATCH_SIZE])
        dev_loader = AmazonDataLoader(dataset.val, model_name, params[C.BATCH_SIZE])
        #test_loader = AmazonDataLoader(dataset.test, model_name, params[C.BATCH_SIZE])

        trainer = Trainer(
            train_loader,
            params,
            dev_loader=dev_loader,
            #test_loader=test_loader,
            random_seed=RANDOM_SEED,
            vocab=dataset.vocab,
            logger=logger,
            resume_training=resume,
            resume_epoch=epoch if resume else None,
            save_dir=save_dir
        )
        trainer.train()

    elif mode in [C.DEV_TYPE, C.TEST_TYPE]:

        # Load saved params and vocabs
        input_path, output_file = args.input_path, args.output_file
        params_filename = '%s/%s' % (input_path, C.SAVED_PARAMS_FILENAME)
        vocab_filename = '%s/%s' % (input_path, C.SAVED_VOCAB_FILENAME)

        logger.log('Loading params..')
        params = json.load(open(params_filename, 'r'))

        logger.log('Loading vocab..')
        vocab = pickle.load(open(vocab_filename, 'rb'))

        model_name = params[C.MODEL_NAME]
        dataset = _get_dataset(
            model_name,
            params[C.CATEGORY],
            params,
            logger
        )

        loader = AmazonDataLoader(
            dataset.val if mode == C.DEV_TYPE else dataset.test,
            model_name,
            params[C.BATCH_SIZE]
        )

        # Load model
        logger.log('Loading saved model..')
        model = LM(
            vocab.get_vocab_size(),
            hsizes(params, model_name),
            params[C.EMBEDDING_DIM],
            params[C.OUTPUT_MAX_LEN],
            params[C.H_LAYERS],
            params[C.DROPOUT],
            params[C.MODEL_NAME]
        )
        use_cuda = torch.cuda.is_available()
        map_location = None if use_cuda else lambda storage, loc: storage # assuming the model was saved from a gpu machine
        model_filename = '%s/%s_%d' % (input_path, C.SAVED_MODEL_FILENAME, epoch)
        model.load_state_dict(torch.load(model_filename, map_location=map_location))

        if use_cuda:
            model.cuda()

        # Instantiate trainer with saved model
        logger.log('Instantiating trainer..')
        trainer = Trainer(
            None,
            params,
            dev_loader=loader,
            #test_loader=loader,
            random_seed=RANDOM_SEED,
            vocab=vocab,
            logger=logger,
            save_dir=save_dir
        )
        logger.log('Adding model to trainer..')
        trainer.model = model

        # Evaluation on test set
        logger.log('Total number of [%s] batches: %d' % (mode.upper(), len(list(loader))))
        trainer.eval(loader, mode, output_filename=output_file)

    else:
        raise 'Unimplemented mode: %s' % mode

def _get_dataset(model, category, params, logger):
    logger.log('Creating dataset for [%s]..' % category.upper())

    base_path = '%s/%s' % (C.BASE_PATH, params[C.CATEGORY])
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    used_params = [params[i] for i in [C.VOCAB_SIZE]]
    filename = '_'.join(list(map(str, [model, category, RANDOM_SEED] + used_params)))
    filename = '%s/%s.pkl' % (
        base_path,
        filename
    )

    if os.path.exists(filename):
        logger.log('Loading dataset from file: %s' % filename)
        with open(filename, 'rb') as fp:
            dataset = pickle.load(fp)
    else:
        dataset = AmazonDataset(params)

        logger.log('Saving dataset in file: %s' % filename)
        with open(filename, 'wb') as fp:
            pickle.dump(dataset, fp, pickle.HIGHEST_PROTOCOL)

    logger.log('Finished loading dataset for [%s] category and [%s] model..' % (category.upper(), model.upper()))
    return dataset

if __name__ == '__main__':
    main()
