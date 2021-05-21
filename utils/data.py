import os
import h5py
import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from conllu import parse_incr
from argparse import ArgumentParser
from transformers import BertTokenizer

def random_lstm(args):
    inp_size = args["featurizer"]["dim"]
    lstm = nn.LSTM(input_size=inp_size, hidden_size=int(inp_size/2),
                num_layers=1, batch_first=True, bidirectional=True)
    for p in lstm.parameters():
        p.requires_grad = False
    return lstm.to(args["device"])
