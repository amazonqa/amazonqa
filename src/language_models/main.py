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
mode = "3"

dataset = AmazonDataset(category, mode)

batch_size = 2
train_loader = AmazonDataLoader(dataset.train[0:4], mode, batch_size)

for batch_idx, data in enumerate(train_loader):
    print(batch_idx, data)

