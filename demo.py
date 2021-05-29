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
cli_args = argp.parse_args()

args = yaml.safe_load(open(cli_args.config_path))

path_to_train = os.path.join(args["corpus"]["corpus_root"], args["corpus"]["train_path"])
path_to_dev = os.path.join(args["corpus"]["corpus_root"], args["corpus"]["dev_path"])
path_to_test = os.path.join(args["corpus"]["corpus_root"], args["corpus"]["test_path"])

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
    test = TwoWordDataset(args, path_to_conllu=path_to_test)

elif task == "parse-depth":
    
    train = OneWordDataset(args=args, path_to_conllu=path_to_train)
    dev = OneWordDataset(args=args, path_to_conllu=path_to_dev)
    loss = L1DepthLoss(args)
    probe = OneWordProbe(args)
    test = OneWordDataset(args, path_to_conllu=path_to_test)

trainer.train_until_convergence(probe=probe, loss=loss, train_loader=train.loader(), dev_loader=dev.loader())

prediction_root = os.path.join(*args["probe"]["predictions_path"].split("/")[:-1])
os.makedirs(prediction_root, exist_ok=True)

params = torch.load(args["probe"]["params_path"], map_location=torch.device('cpu'))
probe.load_state_dict(params)
predictions = predict(probe, test)
torch.save(predictions, args["probe"]["predictions_path"])

reporting_root = args["reporting"]["reporting_root"]
os.makedirs(reporting_root, exist_ok=True)
