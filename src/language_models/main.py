"""Run different seq2seq models
"""

import argparse

from language_models import utils
from language_models.trainer import Trainer
from language_models.dataloader import AmazonDataLoader
from language_models.dataset import AmazonDataset
import constants as C

RANDOM_SEED = 1

def main():
    model, mode = _params()
    params = utils.get_model_params(model)

    dataset = AmazonDataset(category, model)

    if MODE == C.TRAIN_TYPE:
        train_loader = AmazonDataLoader(
            data_type,
            model_type,
            params,
            random_seed=RANDOM_SEED
        )
        dev_loader = AmazonDataLoader(
            data_type,
            model_type,
            params,
            src_vocab=train_loader.src_vocab,
            dst_vocab=train_loader.src_vocab,
            random_seed=RANDOM_SEED
        )
        trainer = Trainer(
            train_loader,
            params,
            dev_loader=dev_loader,
            random_seed=RANDOM_SEED,
            vocab=vocab
        )
        loss = trainer.train()
    else:
        pass

	train_loader = AmazonDataLoader(dataset.train, model, batch_size)
	val_loader = AmazonDataLoader(dataset.val, model, batch_size)
	test_loader = AmazonDataLoader(dataset.test, model, batch_size)


def _params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', type=str, default=C.LM_ANSWERS)
    parser.add_argument('--mode', dest='mode', type=str, default=C.TRAIN_TYPE)
    args = parser.parse_args()
    return args.model_type, args.mode

if __name__ == '__main__':
    main()