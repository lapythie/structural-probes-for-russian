"""Currently expects only UD SynTagRus, 768-dimensional BERT-like encoder"""
"""and only one layer of embeddings to be returned"""

import os
import h5py
import numpy as np
from tqdm.auto import tqdm
from conllu import parse_incr
from argparse import ArgumentParser
from transformers import BertTokenizer
from bert_serving.client import BertClient

argp = ArgumentParser()
argp.add_argument("--bert_dir", default=None, type=str, help="path to directory with a BERT-like model")
argp.add_argument("--bert_alias", default=None, type=str, help="alias of a BERT-like model")
argp.add_argument("--conllu_dir", default=None, type=str, help="directory with train, dev and test parts of a conllu UD dataset")
cli_args = argp.parse_args()

path = "./embeddings/"
if not os.path.isdir(path):
    os.mkdir(path)
if not os.path.isdir(path+cli_args.bert_alias):
    os.mkdir(path+cli_args.bert_alias)

bc = BertClient()
FEATURE_COUNT = 768
FEATURIZER = "bert"
assert len(bc.server_status['pooling_layer']) == 1
LAYER_COUNT = len(bc.server_status['pooling_layer'])
tokenizer = BertTokenizer.from_pretrained(cli_args.bert_dir)
LAYER_NAME = "_".join([str(layer) for layer in bc.server_status['pooling_layer']])

conllus = [fname for fname in os.listdir(cli_args.conllu_dir) if fname.endswith(".conllu")]
embeddings_path = path+cli_args.bert_alias+"/"+cli_args.bert_alias+"_"+LAYER_NAME+".hdf5"
with h5py.File(embeddings_path, "a") as f:
    for fname in conllus:
        split_ = [sp for sp in {"train", "dev", "test"} if sp in fname][0]
        data_file = open(cli_args.conllu_dir+"/"+fname, encoding="utf-8")
        for sent_id, tokenlist in tqdm(enumerate(parse_incr(data_file)), desc="[embedding "+split_+" dataset]"):
            sent = [t["form"] for t in tokenlist if type(t["id"]) == int]
            wordpiece_tokens = [subt for t in sent for subt in tokenizer.wordpiece_tokenizer.tokenize(t)]
            enc = bc.encode([wordpiece_tokens], is_tokenized=True, show_tokens=True)
            embs, server_tokens = enc[0][0][1:-1], enc[1][0] # cut off special tokens embeddings
            assert server_tokens[1:-1] == wordpiece_tokens # make sure server respects tokenization
            dset_name = "/".join([split_, FEATURIZER, LAYER_NAME, str(sent_id)])
            dset = f.create_dataset(dset_name, (LAYER_COUNT, len(wordpiece_tokens), FEATURE_COUNT))
            dset[:,:,:] = embs
