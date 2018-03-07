import matplotlib
matplotlib.use('Agg')
import pickle as pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from helper import *
from model import *
from vocabulary import *

from tqdm import tqdm

import os
import string
import sys

use_cuda = torch.cuda.is_available()

def testIters(encoder, decoder):
    start = time.time()

    criterion = nn.NLLLoss()

    total_loss = 0.0
    num_iters = len(test_pairs)
    for iter in tqdm(range(1, num_iters+1)):
        test_pair = test_pairs[iter - 1]
        input_variable = test_pair[0]
        target_variable = test_pair[1]

        encoder = torch.load()
        decoder = torch.load()
        loss = test(input_variable, target_variable, encoder, decoder, criterion)

        total_loss += loss

MAX_LENGTH=10000


def test(input_variable, target_variable, encoder, decoder, \
        criterion, max_length=MAX_LENGTH):

    loss, target_length = forward(input_variable, target_variable, encoder, \
        decoder, criterion, max_length=MAX_LENGTH)

    return loss.data[0] / target_length

def forward(input_variable, target_variable, encoder, decoder, encoder_optimizer, \
        decoder_optimizer, criterion, max_length=MAX_LENGTH):

    encoder_hidden = encoder.init_hidden()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    # Without teacher forcing: use its own predictions as the next input
    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        loss += criterion(decoder_output, target_variable[di])
        if ni == EOS_token: break
    return loss, target_length


def normalize_pair(triplet):
    pair = [triplet[0], triplet[1]]
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    return [val.lower().translate(table).strip() for val in pair]

def prepareData(category):
    test_data = pickle.load(open('../../data/nn/' + category + '_qar_test.pickle', 'rb'))

    test_pairs  = [normalize_pair(triplet) for triplet in test_data]

    input_lang = Vocabulary(40000)
    output_lang = Vocabulary(60000)

    print("Creating languages...")
    for pair in test_pairs:
        input_lang.add_sequence(pair[0].split())
        output_lang.add_sequence(pair[1].split())

    input_lang.trim()
    output_lang.trim()

    print("Counted words:")
    print("qlang: ", input_lang._num_tokens)
    #print(input_lang._token2index)
    print("alang: ", output_lang._num_tokens)
    #print(output_lang._token2index)

    test_pairs = [variablesFromPair(input_lang, output_lang, pair) \
            for pair in test_pairs]

    print("Test Pairs:", len(test_pairs))

    return input_lang, output_lang, test_pairs


category = sys.argv[1]
run = sys.argv[2]

input_lang, output_lang, test_pairs = prepareData(category)

#hidden_size = 256
#encoder_ = EncoderRNN(input_lang._num_tokens + input_lang._num_reserved, hidden_size)
#decoder_ = DecoderRNN(hidden_size, output_lang._num_tokens + output_lang._num_reserved)

if use_cuda:
    encoder_ = encoder_.cuda()
    decoder_ = decoder_.cuda()

testIters(encoder_, decoder_)
