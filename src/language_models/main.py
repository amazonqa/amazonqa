# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

from dataset import AmazonDataset
from dataloader import AmazonDataLoader


category = "Video_Games"
mode = "1"

dataset = AmazonDataset(category, mode)

batch_size = 100
train_loader = AmazonDataLoader(dataset.train, mode, batch_size)
val_loader = AmazonDataLoader(dataset.val, mode, batch_size)
test_loader = AmazonDataLoader(dataset.test, mode, batch_size)

for batch_idx, data in enumerate(test_loader):
    paded_answers = data
    print(batch_idx)

