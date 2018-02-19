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

from encoder import *
from decoder import *
from helper import *
from lang import *
import os
use_cuda = torch.cuda.is_available()

def trainIters(encoder, decoder, epochs, num_iters, learning_rate):
    start = time.time()
    plot_losses = []
    loss_total = 0  # Reset every print_every

    #enocder_optimizer1 = optim.SGD(encoder.parameters(), lr=learning_rate)
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()
    import sys
    run = sys.argv[1]
    for epoch in range(epochs):
        print('Epoch ', epoch, ' starting\n')
        training_pairs = [random.choice(pairs) for i in range(num_iters)]
        total_loss = 0.0
        for iter in range(1, num_iters+1):
            training_pair = training_pairs[iter - 1]
            input_variable = training_pair[0]
            target_variable = training_pair[1]

            loss = train(input_variable, target_variable, encoder, decoder, \
                    encoder_optimizer, decoder_optimizer, criterion)

            loss_total += loss

            print_every = 5000
            if iter % print_every == 0 or iter == num_iters - 1:
                loss_avg = loss_total / iter
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / num_iters), \
                    iter, iter / num_iters * 100, loss_avg))

        print("SAVING MODELS FOR EPOCH - ", str(epoch))
        dir = 'model' + run
        if not os.path.exists(dir):
            os.makedirs(dir)

        torch.save(encoder, dir + 'encoder_%d')
        torch.save(decoder, dir + 'decoder_%d')

    #FIXME put test stuff
teacher_forcing_ratio = 0.0
MAX_LENGTH=10000

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, \
        decoder_optimizer, criterion, max_length=MAX_LENGTH):

    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

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

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def normalize_pair(triplet):
    pair = [triplet[1], triplet[2]]
    return np.array([val.lower().strip() for val in pair])

def prepareData(qlang, alang):
    data = np.array(np.load('../../data/qa_reviews.npy'))
    pairs = [normalize_pair(triplet) for triplet in data]

    input_lang = Lang(qlang)
    output_lang = Lang(alang)

    print("Creating languages...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    print("Total Pairs:", len(pairs))

    return input_lang, output_lang, pairs, len(pairs)


input_lang, output_lang, raw_pairs, l = prepareData('qlang', 'alang')

pairs = [variablesFromPair(input_lang, output_lang, pair) for pair in raw_pairs]

import random
random.shuffle(pairs)

num_train = int(l*0.7)
num_test = l - num_train

training_pairs = pairs[0:num_train]
test_pairs = pairs[num_train:]
pickle.dump(test_pairs, open('test_pairs.pickle', 'wb'))

hidden_size = 256
encoder_ = EncoderRNN(input_lang.n_words, hidden_size)
decoder_ = DecoderRNN(hidden_size, output_lang.n_words)

if use_cuda:
    encoder_ = encoder_.cuda()
    decoder_ = decoder_.cuda()

trainIters(encoder_, decoder_, 10, l, 0.01)
