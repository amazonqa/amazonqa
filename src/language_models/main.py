"""Run different seq2seq models
"""

import os
import argparse
import pickle

from language_models import utils
from language_models.trainer import Trainer
from language_models.dataloader import AmazonDataLoader
from language_models.dataset import AmazonDataset
import constants as C
from logger import Logger

RANDOM_SEED = 1

def main():
    args = _params()
    model_name, mode = args.model_name, args.mode
    params = utils.get_model_params(model_name)
    params[C.MODEL_NAME] = model_name
    category = params[C.CATEGORY]

    logger = Logger()
    params[C.LOG_FILENAME] = logger.logfilename

    logger.log('\n Model: %s, Mode = %s, Category = %s \n' % (model_name, mode, category))
    dataset = _get_dataset(model_name, category, params, logger)

    if mode == C.TRAIN_TYPE:
        train_loader = AmazonDataLoader(dataset.train, model_name, params[C.BATCH_SIZE])
        dev_loader = AmazonDataLoader(dataset.val, model_name, params[C.BATCH_SIZE])

        #for batch_idx, data in enumerate(train_loader):
        #    answs, lengths = data
        #    print(dataset.vocab.token_list_from_indices(answs[0]), lengths[0])

        trainer = Trainer(
            train_loader,
            params,
            dev_loader=dev_loader,
            random_seed=RANDOM_SEED,
            vocab=dataset.vocab,
            logger=logger
        )
        trainer.train()
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
        dataset = AmazonDataset(category, model, params[C.VOCAB_SIZE])

        logger.log('Saving dataset in file: %s' % filename)
        with open(filename, 'wb') as fp:
            pickle.dump(dataset, fp, pickle.HIGHEST_PROTOCOL)

    logger.log('Finished loading dataset for [%s] category and [%s] model..' % (category.upper(), model.upper()))
    return dataset

def _params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', dest='model_name', type=str, default=C.LM_ANSWERS)
    parser.add_argument('--mode', dest='mode', type=str, default=C.TRAIN_TYPE)
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    main()
