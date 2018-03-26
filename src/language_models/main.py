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
    model, mode, category = _params()
    print(model, mode, category)
    params = utils.get_model_params(model)

    dataset = AmazonDataset(category, model, params[C.VOCAB_SIZE])

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
        pass

def _params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', type=str, default=C.LM_ANSWERS)
    parser.add_argument('--mode', dest='mode', type=str, default=C.TRAIN_TYPE)
    parser.add_argument('--category', dest='category', type=str, default=C.VIDEO_GAMES)
    args = parser.parse_args()
    return args.model, args.mode, args.category

if __name__ == '__main__':
    main()