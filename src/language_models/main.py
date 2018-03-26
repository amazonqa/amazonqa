# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

from dataset import AmazonDataset
from dataloader import AmazonDataLoader
import constants as C


category = C.VIDEO_GAMES
model = C.LM_QUESTION_ANSWERS_REVIEWS

dataset = AmazonDataset(category, model)

batch_size = 100
train_loader = AmazonDataLoader(dataset.train, model, batch_size)
val_loader = AmazonDataLoader(dataset.val, model, batch_size)
test_loader = AmazonDataLoader(dataset.test, model, batch_size)

for batch_idx, data in enumerate(test_loader):
    paded_answers, padded_questions = data
    print(batch_idx)

