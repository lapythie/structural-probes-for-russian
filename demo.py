import os
import h5py
import yaml
import torch
import numpy as np
import torch.nn as nn
import networkx as nx
from tqdm.auto import tqdm
from conllu import parse_incr
from collections import namedtuple
from argparse import ArgumentParser
from transformers import BertTokenizer

from utils.data import *
from utils.loss import *
from utils.probe import *
from utils.training import ProbeTrainer, predict

argp = ArgumentParser()
argp.add_argument("--config_path", default=None, type=str, help="path to yaml config file")
argp.add_argument("--conllu_dir", default=None, type=str, help="directory with train, dev and test parts of a conllu UD dataset")
cli_args = argp.parse_args()

path_to_train = cli_args.conllu_dir+"/"+[p for p in os.listdir(cli_args.conllu_dir)
                                         if p.endswith(".conllu") and "train" in p][0]
path_to_dev = cli_args.conllu_dir+"/"+[p for p in os.listdir(cli_args.conllu_dir)
                                       if p.endswith(".conllu") and "dev" in p][0]

args = yaml.safe_load(open(cli_args.config_path))
task = args["probe"]["task"]

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

trainer = ProbeTrainer(args)

predictor_root = os.path.join(*trainer.probe_params_path.split("/")[:-1])
os.makedirs(predictor_root, exist_ok=True)

if task == "parse-distance":
    train = TwoWordDataset(args=args, path_to_conllu=path_to_train)
    dev = TwoWordDataset(args=args, path_to_conllu=path_to_dev)
    loss = L1DistanceLoss(args)
    probe = TwoWordProbe(args)
elif task == "parse-depth":
    train = OneWordDataset(args=args, path_to_conllu=path_to_train)
    dev = OneWordDataset(args=args, path_to_conllu=path_to_dev)
    loss = L1DepthLoss(args)
    probe = OneWordProbe(args)

trainer.train_until_convergence(probe=probe, loss=loss, train_loader=train.loader(), dev_loader=dev.loader())
