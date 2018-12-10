#!python3

"""
Training script: load a config file, create a new model using it,
then train that model.
"""
import json
import yaml
import argparse
import os.path

import numpy as np
import torch
import h5py

import constants as C
from bidaf import BidafModel
from logger import Logger

from dataset import load_data, tokenize_data, EpochGen
from dataset import SymbolEmbSourceNorm
from dataset import SymbolEmbSourceText
from dataset import symbol_injection


def try_to_resume(logger, exp_folder):
    if os.path.isfile(exp_folder + '/checkpoint'):
        logger.log('Trying to resume from checkpoints...')
        checkpoint = h5py.File(exp_folder + '/checkpoint')
    else:
        logger.log('No checkpoint found.')
        checkpoint = None
    return checkpoint


def reload_state(logger, checkpoint, config, args):
    """
    Reload state before predicting.
    """
    logger.log('Loading model from checkpoint...')
    model, id_to_token, id_to_char = BidafModel.from_checkpoint(
        config['bidaf'], checkpoint, args.epoch)

    token_to_id = {tok: id_ for id_, tok in id_to_token.items()}
    char_to_id = {char: id_ for id_, char in id_to_char.items()}

    len_tok_voc = len(token_to_id)
    len_char_voc = len(char_to_id)

    logger.log('Loading data...')
    with open(args.data) as f_o:
        data, _ = load_data(args.data)
    limit_passage = config.get('training', {}).get('limit')
    data = tokenize_data(logger, data, token_to_id, char_to_id, None, False, limit_passage)

    logger.log('Tokenizing data...')
    id_to_token = {id_: tok for tok, id_ in token_to_id.items()}
    id_to_char = {id_: char for char, id_ in char_to_id.items()}

    logger.log('Creating dataloader...')
    data = get_loader(data, args)

    if len_tok_voc != len(token_to_id):
        need = set(tok for id_, tok in id_to_token.items()
                   if id_ >= len_tok_voc)

        if args.word_rep:
            with open(args.word_rep) as f_o:
                pre_trained = SymbolEmbSourceText(
                    f_o, need)
        else:
            pre_trained = SymbolEmbSourceText([], need)

        cur = model.embedder.embeddings[0].embeddings.weight.data.numpy()
        mean = cur.mean(0)
        if args.use_covariance:
            cov = np.cov(cur, rowvar=False)
        else:
            cov = cur.std(0)

        rng = np.random.RandomState(2)
        oovs = SymbolEmbSourceNorm(mean, cov, rng, args.use_covariance)

        if args.word_rep:
            logger.log('Augmenting with pre-trained embeddings...')
        else:
            logger.log('Augmenting with random embeddings...')

        model.embedder.embeddings[0].embeddings.weight.data = torch.from_numpy(
            symbol_injection(
                id_to_token, len_tok_voc,
                model.embedder.embeddings[0].embeddings.weight.data.numpy(),
                pre_trained, oovs))

    if len_char_voc != len(char_to_id):
        logger.log('Augmenting with random char embeddings...')
        pre_trained = SymbolEmbSourceText([], None)
        cur = model.embedder.embeddings[1].embeddings.weight.data.numpy()
        mean = cur.mean(0)
        if args.use_covariance:
            cov = np.cov(cur, rowvar=False)
        else:
            cov = cur.std(0)

        rng = np.random.RandomState(2)
        oovs = SymbolEmbSourceNorm(mean, cov, rng, args.use_covariance)

        model.embedder.embeddings[1].embeddings.weight.data = torch.from_numpy(
            symbol_injection(
                id_to_char, len_char_voc,
                model.embedder.embeddings[1].embeddings.weight.data.numpy(),
                pre_trained, oovs))

    if torch.cuda.is_available() and args.cuda:
        model.cuda()
    model.eval()

    return model, id_to_token, id_to_char, data


def get_loader(data, args):
    data = EpochGen(
        data,
        batch_size=args.batch_size,
        shuffle=False)
    return data


def predict(logger, model, data):
    """
    Train for one epoch.
    """
    for batch_id, (qids, passages, queries, targets, _, _) in enumerate(data):
        decoder_outputs, decoder_hidden, ret_dict = model(
            passages[:2], passages[2],
            queries[:2], queries[2],
            targets[0], 0
        )

        output_seq = ret_dict['sequence']
        output_lengths = ret_dict['length']
        output_seq = torch.cat(output_seq, 1)

        output_seq = output_seq.data.cpu().numpy()
        for seq_itr, length in enumerate(output_lengths):
            length = int(length)
            seq = output_seq[seq_itr, :length]
            if seq[-1] == C.EOS_INDEX:
                seq = seq[:-1]
            
            yield (qids[seq_itr], seq)
            # tokens = self.vocab.token_list_from_indices(seq)
            # generated_answer = ' '.join(tokens)
            # fp.write(generated_answer + '\n')
            
            # gold_answers = []
            # question_id = question_ids[seq_itr]
            # answer_ids = dataloader.questionAnswersDict[question_id]
            # for answer_id in answer_ids:
            #     answer_seq = dataloader.answersDict[answer_id]
            #     answer_tokens = self.vocab.token_list_from_indices(answer_seq)
            #     gold_answers.append(' '.join(answer_tokens))

            # gold_answers_dict[question_id] = gold_answers
            # generated_answer_dict[question_id] = [generated_answer]


        # predictions = model.get_best_span(start_log_probs, end_log_probs)
        # predictions = predictions.cpu()
        # passages = passages[0].cpu().data
        # for qid, mapping, tokens, pred in zip(
        #         qids, mappings, passages, predictions):
        #     yield (qid, tokens[pred[0]:pred[1]],
        #            mapping[pred[0], 0],
        #            mapping[pred[1]-1, 1])
    return


def main():
    """
    Main prediction program.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument("exp_folder", help="Experiment folder")
    argparser.add_argument("data", help="Prediction data")
    argparser.add_argument("dest", help="Write predictions in")
    argparser.add_argument("--epoch",
                           type=int, default=None,
                           help="Epoch to load model from")
    argparser.add_argument("--word_rep",
                           help="Text file containing pre-trained "
                           "word representations.")
    argparser.add_argument("--batch_size",
                           type=int, default=64,
                           help="Batch size to use")
    argparser.add_argument("--cuda",
                           type=bool, default=torch.cuda.is_available(),
                           help="Use GPU if possible")
    argparser.add_argument("--use_covariance",
                           action="store_true",
                           default=False,
                           help="Do not assume diagonal covariance matrix "
                           "when generating random word representations.")

    args = argparser.parse_args()
    logger = Logger()

    config_filepath = os.path.join(args.exp_folder, 'config.yaml')
    with open(config_filepath) as f:
        config = yaml.load(f)

    checkpoint = try_to_resume(logger, args.exp_folder)

    if checkpoint:
        model, id_to_token, id_to_char, data = reload_state(logger,
            checkpoint, config, args)
    else:
        logger.log('Need a valid checkpoint to predict.')
        return

    if torch.cuda.is_available() and args.cuda:
        data.tensor_type = torch.cuda.LongTensor

    with open(args.dest, 'w') as f_o:
        for qid, toks in predict(logger, model, data):
            toks = ' '.join(id_to_token[tok] for tok in toks)
            print(repr(qid), repr(toks), file=f_o)

    return


if __name__ == '__main__':
    main()
