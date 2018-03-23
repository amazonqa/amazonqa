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

test_dataset = AmazonDataset(category, mode, 'test')

batch_size = 2
test_loader = AmazonDataLoader(test_dataset, mode, batch_size)

for batch_idx, data in enumerate(test_loader):
    print(batch_idx, data)

