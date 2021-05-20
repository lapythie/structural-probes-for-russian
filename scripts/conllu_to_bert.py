"""Currently expects only UD SynTagRus and only one layer of embeddings to be returned"""

import os
from argparse import ArgumentParser
import numpy as np
from conllu import parse_incr
import h5py
from transformers import BertTokenizer

if __name__ == "__main__":
    argp = ArgumentParser()
    argp.add_argument("--bert_dir", default=None, type=str,
                      help="path to directory with a BERT-like model")
    argp.add_argument("--bert_alias", default=None, type=str,
                      help="alias of a BERT-like model")
    argp.add_argument("--conllu_dir", default=None, type=str,
                      help="directory with train, dev and test parts of a conllu UD dataset")
    cli_args = argp.parse_args()

    tokenizer = BertTokenizer.from_pretrained(cli_args.bert_dir)

    path = "./embeddings/"
    if not os.path.isdir(path):
        os.mkdir(path)
    if not os.path.isdir(path+cli_args.bert_alias):
        os.mkdir(path+cli_args.bert_alias)

    conllu_filenames = [fname for fname in os.listdir(cli_args.conllu_dir)
                        if fname.endswith(".conllu")]

    FEATURE_COUNT = 768
    FEATURIZER = "bert"
    
    for fname in conllu_filenames:
            
        split_ = [split_ for split_ in {"train", "dev", "test"} if split_ in fname][0]
        print(split_)
        data_file = open(cli_args.conllu_dir+"/"+fname, encoding="utf-8")
        for sent_id, tokenlist in enumerate(parse_incr(data_file)):
            sent = [t["form"] for t in tokenlist if type(t["id"]) == int]
            wordpiece_tokens = [subt for t in sent for subt
                                in tokenizer.wordpiece_tokenizer.tokenize(t)]
