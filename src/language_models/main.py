"""Run different seq2seq models
"""

import os
import argparse
import pickle

from language_models import utils
from language_models.trainer import Trainer, hsizes
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
        test_loader = AmazonDataLoader(dataset.test, model_name, params[C.BATCH_SIZE])

        #for batch_idx, data in enumerate(train_loader):
        #    answs, lengths = data
        #    print(dataset.vocab.token_list_from_indices(answs[0]), lengths[0])

        trainer = Trainer(
            train_loader,
            params,
            dev_loader=dev_loader,
            dev_loader=test_loader,
            random_seed=RANDOM_SEED,
            vocab=dataset.vocab,
            logger=logger
        )
        trainer.train()

    elif mode in [C.DEV_TYPE, C.TEST_TYPE]:

        # Load saved params and vocabs
        input_path, epoch, output_file = args.input_path, args.epoch, args.output_file
        params_filename = '%s/%s' % (input_path, C.SAVED_PARAMS_FILENAME)
        vocab_filename = '%s/%s' % (input_path, C.SAVED_VOCAB_FILENAME)

        logger.log('Loading params..')
        params = json.load(open('%s/params.json' % input_path, 'r'))

        logger.log('Loading vocab..')
        vocab = pickle.load(open('%s/%s' % input_path, 'rb'))

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
            hsizes(params[C.HDIM], model_name),
            params[C.EMBEDDING_DIM],
            params[C.OUTPUT_MAX_LEN],
            params[C.H_LAYERS],
            params[C.DROPOUT],
            params[C.MODEL_NAME]
        )
        use_cuda = torch.cuda.is_available()
        map_location = None if use_cuda else 'cpu' # assuming the model was saved from a gpu machine
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
            test_loader=loader,
            random_seed=RANDOM_SEED
        )
        logger.log('Adding model to trainer..')
        trainer.model = model

        # Evaluation on test set
        logger.log('Total number of [%s] batches: %d' % len(list(loader)))
        trainer.eval(test_loader, mode, output_filename=output_file)

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
