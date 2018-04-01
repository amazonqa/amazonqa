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
    model, mode, category = _params()
    params = utils.get_model_params(model)

    logger = Logger()
    logger.log('\n Model: %s, Mode = %s, Category = %s \n' % (model, mode, category))
    dataset = _get_dataset(model, category, params, logger)

    if mode == C.TRAIN_TYPE:
        train_loader = AmazonDataLoader(dataset.train, model, params[C.BATCH_SIZE])
        dev_loader = AmazonDataLoader(dataset.val, model, params[C.BATCH_SIZE])

        #for batch_idx, data in enumerate(train_loader):
        #    answs, lengths = data
        #    print(dataset.vocab.token_list_from_indices(answs[0]), lengths[0])

        trainer = Trainer(
            train_loader,
            params,
            dev_loader=dev_loader,
            random_seed=RANDOM_SEED,
            vocab=dataset.vocab
        )
        trainer.train()
    else:
        raise 'Unimplemented mode: %s' % mode

def _get_dataset(model, category, params, logger):
    logger.log('Creating dataset for [%s]..' % category.upper())

    if not os.path.exists(C.BASE_PATH):
        os.makedirs(C.BASE_PATH)

    used_params = [params[i] for i in [C.VOCAB_SIZE]]
    filename = '_'.join(list(map(str, [model, category, RANDOM_SEED] + used_params)))
    filename = '%s/%s/%s.pkl' % (
        C.BASE_PATH,
        params[C.MODEL_NAME],
        filename
    )

    if os.path.exists(filename):
        logger.log('Loading dataset from file: %s' % filename)
        with open(filename, 'rb') as fp:
            loader = pickle.load(fp)
    else:
        dataset = AmazonDataset(category, model, params[C.VOCAB_SIZE])

        logger.log('Saving dataset in file: %s' % filename)
        with open(filename, 'wb') as fp:
            pickle.dump(loader, fp, pickle.HIGHEST_PROTOCOL)

    logger.log('Finished loading dataset for [%s] category and [%s] model..' % (category.upper(), model.upper()))
    return dataset

def _params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', type=str, default=C.LM_ANSWERS)
    parser.add_argument('--mode', dest='mode', type=str, default=C.TRAIN_TYPE)
    parser.add_argument('--category', dest='category', type=str, default=C.VIDEO_GAMES)
    parser.add_argument('--epoch', dest='epoch', type=int, default=0)
    args, _ = parser.parse_known_args()
    return args.model, args.mode, args.category

if __name__ == '__main__':
    main()
