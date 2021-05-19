"""Currently expects only UD SynTagRus and only one layer of embeddings to be returned"""

import os
from argparse import ArgumentParser
import numpy as np
from conllu import parse_incr
import h5py
from transformers import BertTokenizer

if __name__ == "__main__":
    argp = ArgumentParser()
    argp.add_argument("--path_to_bert", default=None, type=str,
                      help="path to directory with a BERT-like model")
    argp.add_argument("--bert_alias", default=None, type=str,
                      help="alias of a BERT-like model")
    cli_args = argp.parse_args()

    tokenizer = BertTokenizer.from_pretrained(cli_args.path_to_bert)

    path = "./embeddings/"
    if not os.path.isdir(path):
        os.mkdir(path)
    if not os.path.isdir(path+cli_args.bert_alias):
        os.mkdir(path+cli_args.bert_alias)
