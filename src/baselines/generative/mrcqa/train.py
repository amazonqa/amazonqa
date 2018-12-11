#!python3
"""
Training script: load a config file, create a new model using it,
then train that model.
"""
import json
import yaml
import argparse
import os.path
import itertools

import numpy as np
import torch
import torch.optim as optim
import h5py
from tqdm import tqdm

import constants as C

from bidaf import BidafModel
from loss import Loss
from logger import Logger

import checkpointing
from dataset import load_data, tokenize_data, EpochGen
from dataset import SymbolEmbSourceNorm
from dataset import SymbolEmbSourceText
from dataset import symbol_injection
from dataset import default_vocab

def try_to_resume(logger, force_restart, exp_folder):
    if force_restart:
        logger.log('Ignoring any checkpoints...')
        return None, None, 0
    elif os.path.isfile(exp_folder + '/checkpoint'):
        logger.log('Trying to resume from checkpoints...')
        checkpoint = h5py.File(exp_folder + '/checkpoint')
        epoch = checkpoint['training/epoch'][()] + 1
        # Try to load training state.
        try:
            checkpoint_file = exp_folder + ('/checkpoint_%d.opt' % epoch)
            training_state = torch.load(checkpoint_file)
            logger.log('Loaded checkpoint from %s' % checkpoint_file)
        except FileNotFoundError:
            logger.log('No checkpoint found.')
            training_state = None
    else:
        return None, None, 0

    return checkpoint, training_state, epoch


def reload_state(logger, checkpoint, training_state, config, args):
    """
    Reload state when resuming training.
    """
    logger.log('Loading model from checkpoint...')
    model, id_to_token, id_to_char = BidafModel.from_checkpoint(
        config['bidaf'], checkpoint)

    if torch.cuda.is_available() and args.cuda:
        model.cuda()

    model.train()

    optimizer = get_optimizer(model, config, training_state)

    token_to_id = {tok: id_ for id_, tok in id_to_token.items()}
    char_to_id = {char: id_ for id_, char in id_to_char.items()}

    len_tok_voc = len(token_to_id)
    len_char_voc = len(char_to_id)

    logger.log('Loading data...')
    data, _ = load_data(args.data)
    limit_passage = config.get('training', {}).get('limit')
    vocab_size = config.get('training', {}).get('vocab_size', None)

    logger.log('Tokenizing data...')
    data = tokenize_data(logger, data, token_to_id, char_to_id, vocab_size, True, limit_passage)

    logger.log('Creating dataloader...')
    data = get_loader(data, config)

    assert len(token_to_id) == len_tok_voc
    assert len(char_to_id) == len_char_voc

    return model, id_to_token, id_to_char, optimizer, data


def get_optimizer(model, config, state):
    """
    Get the optimizer
    """
    parameters = filter(lambda p: p.requires_grad,
                        model.parameters())
    optimizer = optim.Adam(
        parameters,
        lr=config['training'].get('lr', 0.01),
        betas=config['training'].get('betas', (0.9, 0.999)),
        eps=config['training'].get('eps', 1e-8),
        weight_decay=config['training'].get('weight_decay', 0))

    if state is not None:
        optimizer.load_state_dict(state)

    return optimizer


def get_loader(data, config):
    data = EpochGen(
        data,
        batch_size=config.get('training', {}).get('batch_size', 32),
        shuffle=True)
    return data


def init_state(logger, config, args):
    logger.log('Loading data...')

    with open(args.data) as f_o:
        data, _ = load_data(args.data)
    
    limit_passage = config.get('training', {}).get('limit')
    vocab_size = config.get('training', {}).get('vocab_size', None)

    logger.log('Tokenizing data...')
    data, token_to_id, char_to_id = tokenize_data(logger, data, vocab_size, True, limit_passage)
    data = get_loader(data, config)

    id_to_token = {id_: tok for tok, id_ in token_to_id.items()}
    id_to_char = {id_: char for char, id_ in char_to_id.items()}

    assert(token_to_id[C.SOS_TOKEN] == C.SOS_INDEX)
    assert(token_to_id[C.UNK_TOKEN] == C.UNK_INDEX)
    assert(token_to_id[C.EOS_TOKEN] == C.EOS_INDEX)
    assert(token_to_id[C.PAD_TOKEN] == C.PAD_INDEX)

    logger.log('Creating model...')
    model = BidafModel.from_config(config['bidaf'], id_to_token, id_to_char)

    if args.word_rep:
        logger.log('Loading pre-trained embeddings...')
        with open(args.word_rep) as f_o:
            pre_trained = SymbolEmbSourceText(
                    f_o,
                    set(tok for id_, tok in id_to_token.items() if id_ != 0))
        mean, cov = pre_trained.get_norm_stats(args.use_covariance)
        rng = np.random.RandomState(2)
        oovs = SymbolEmbSourceNorm(mean, cov, rng, args.use_covariance)

        model.embedder.embeddings[0].embeddings.weight.data = torch.from_numpy(
            symbol_injection(
                id_to_token, 0,
                model.embedder.embeddings[0].embeddings.weight.data.numpy(),
                pre_trained, oovs))
    else:
        pass  # No pretraining, just keep the random values.

    # Char embeddings are already random, so we don't need to update them.

    if torch.cuda.is_available() and args.cuda:
        model.cuda()

    model.train()

    optimizer = get_optimizer(model, config, state=None)
    return model, id_to_token, id_to_char, optimizer, data


def train_epoch(loss, model, optimizer, data, args, logger, teacher_forcing_ratio):
    """
    Train for one epoch.
    """
    batch_skipped = 0
    for batch_id, (qids, passages, queries, targets, _, _) in enumerate(data):
        print(batch_id)

        try:
            outputs, _, _ = model(
                passages[:2], passages[2],
                queries[:2], queries[2],
                targets[0],
                teacher_forcing_ratio
            )
        except:
            print("Skipping batch:", batch_id)
            batch_skipped += 1
            continue

        # loss and gradient computation
        batch_loss, batch_perplexity = loss.eval_batch_loss(outputs, targets[0])

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        batch_loss = batch_loss.data.item()
        if batch_id % 100 == 0:
            logger.log('\n\tMean [TRAIN] Loss for batch %d = %.2f' % (batch_id, batch_loss))
            logger.log('\tMean [TRAIN] Perplexity for batch %d = %.2f' % (batch_id, batch_perplexity))

    return loss


def main():
    """
    Main training program.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument("exp_folder", help="Experiment folder")
    argparser.add_argument("data", help="Training data")
    argparser.add_argument("--force_restart",
                           action="store_true",
                           default=False,
                           help="Force restart of experiment: "
                           "will ignore checkpoints")
    argparser.add_argument("--word_rep",
                           help="Text file containing pre-trained "
                           "word representations.")
    argparser.add_argument("--cuda",
                           type=bool, default=torch.cuda.is_available(),
                           help="Use GPU if possible")
    argparser.add_argument("--use_covariance",
                           action="store_true",
                           default=False,
                           help="Do not assume diagonal covariance matrix "
                           "when generating random word representations.")

    args = argparser.parse_args()

    config_filepath = os.path.join(args.exp_folder, 'config.yaml')
    with open(config_filepath) as f:
        config = yaml.load(f)

    logger = Logger()

    checkpoint, training_state, epoch = try_to_resume(logger,
            args.force_restart, args.exp_folder)


    if checkpoint:
        logger.log('Resuming training...')
        model, id_to_token, id_to_char, optimizer, data = reload_state(logger, checkpoint, training_state, config, args)
    else:
        logger.log('Preparing to train...')
        model, id_to_token, id_to_char, optimizer, data = init_state(logger, 
            config, args)

        #checkpoint = h5py.File(os.path.join(args.exp_folder, 'checkpoint'))

        #logger.log('Saving vocab...')
        #checkpointing.save_vocab(checkpoint, 'vocab', id_to_token)
        #checkpointing.save_vocab(checkpoint, 'c_vocab', id_to_char)

    if torch.cuda.is_available() and args.cuda:
        data.tensor_type = torch.cuda.LongTensor

    print("TENSOR TYPE:", data.tensor_type)

    train_for_epochs = config.get('training', {}).get('epochs')
    teacher_forcing_ratio = config.get('training', {}).get('teacher_forcing_ratio', 1.0)
    if train_for_epochs is not None:
        epochs = range(epoch, train_for_epochs)
    else:
        epochs = itertools.count(epoch)

    loss = Loss()
    best_epoch = 0
    losses, perplexities = [], []
    min_loss, min_perplexity, best_epoch = np.nan, np.nan, 0
    for epoch in epochs:
        loss.reset()

        logger.log('\n  --- STARTING EPOCH : %d --- \n' % epoch)
        train_epoch(loss, model, optimizer, data, args, logger, teacher_forcing_ratio)
        logger.log('\n  --- END OF EPOCH : %d --- \n' % epoch)

        epoch_loss = loss.epoch_loss()
        epoch_perplexity = loss.epoch_perplexity()
        losses.append(epoch_loss)
        perplexities.append(epoch_perplexity)
        min_loss, min_perplexity = np.nanmin(losses), np.nanmin(perplexities)
        if min_loss == epoch_loss:
            best_epoch = epoch

        mode = 'train'.upper()
        logger.log('\n\tEpoch [%s] Loss = %.4f, Min [%s] Loss = %.4f' % (mode, epoch_loss, mode, min_loss))
        logger.log('\tEpoch [%s] Perplexity = %.2f, Min [%s] Perplexity = %.2f' % (mode, epoch_perplexity, mode, min_perplexity))
        logger.log('\tBest Epoch = %d' % (best_epoch))

        # Compute epoch loss and perplexity
        checkpointing.checkpoint(
            model,
            epoch,
            best_epoch,
            optimizer,
            checkpoint,
            args.exp_folder
        )

    return


if __name__ == '__main__':
    main()
