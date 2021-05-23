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

def random_lstm(args):
    inp_size = args["featurizer"]["dim"]
    lstm = nn.LSTM(input_size=inp_size, hidden_size=int(inp_size/2),
                num_layers=1, batch_first=True, bidirectional=True)
    for p in lstm.parameters():
        p.requires_grad = False
    return lstm.to(args["device"])

class ProbingDataset(torch.utils.data.Dataset):
    """Reads conllu files

    Attributes:
        args: config dictionary from a yaml file
    """
    def __init__(self, args, path_to_conllu, cached_labels):
        self.args=args
        self.path_to_conllu = path_to_conllu
        self.cached_labels = cached_labels
        self.path_to_embeddings = args["featurizer"]["path_to_embeddings"]
        self.batch_size = args["probe_training"]["batch_size"]
        self.featurizer_type = args["featurizer"]["featurizer_type"]
        self.split_ = [sp for sp in {"dev", "test", "train"} if sp in self.path_to_conllu][0]
        self.Sentence = namedtuple(typename="Sentence", field_names=["tokens", "length", "tree", "embeddings"])
        self.trees, self.roots, self.sents, self.lengths  = self.trees_sents_roots_lengths()
        self.layer = args["featurizer"]["layer"]
        self.path_to_tokenizer = args["featurizer"]["path_to_tokenizer"]
        if self.path_to_tokenizer:
            self.data_file = h5py.File(self.path_to_embeddings, mode="r")
            self.tokenizer = BertTokenizer.from_pretrained(self.path_to_tokenizer)
            self.subtokens, self.spans = self.wordpiece_tokenize()
        else:
            self.projection = args["featurizer"]["projection"]
            self.data_file = h5py.File(f"{self.path_to_embeddings}/elmo-{self.split_}.hdf5", mode="r")
            if args["featurizer"]["decay"]:
                self.length_to_decay, self.length_to_norm = self.compute_decays_and_norms()

    def read_embeddings(self, index):
        """Reads embeddings from disk one sentence at a time, averages subword embeddings for a token"""
        if self.path_to_tokenizer:
            sentence_embeddings = []
            dset = self.data_file["/".join(self.split_, "bert", self.layer, str(index))][...][0]
            for span in self.spans[index]:
                token_embeddings = dset[slice(*span)]
                averaged_token_embedding = token_embeddings.mean(axis=0)
                sentence_embeddings.append(averaged_token_embedding)
            sentence_embeddings = np.vstack(sentence_embeddings)
        else: # ELMo it is then
            dset = self.data_file[f"{self.split_}/elmo/{index}"][...]
            sentence_embeddings = dset[int(self.layer)]
            if self.args["featurizer"]["decay"]:
                sentence_embeddings = self.decay(sentence_embeddings, index)
            elif self.projection:
                with torch.no_grad():
                    proj, _ = self.projection(torch.tensor(sentence_embeddings, device=args["device"]).unsqueeze(0))
                sentence_embeddings = proj.squeeze(0)
        return sentence_embeddings


    def trees_sents_roots_lengths(self):
        """Precomputes and caches dependency trees and their roots"""
        trees = []
        sents = []
        roots = []
        lengths = []
        with open(self.path_to_conllu, encoding="utf-8") as data_file:
            for tokenlist in tqdm(parse_incr(data_file), desc="[building trees]"):
                tokens = [t["form"] for t in tokenlist if type(t["id"]) == int]
                sents.append(tokens)
                lengths.append(len(tokens))
                G = nx.Graph()
                G.add_edges_from([(t["id"], t["head"]) for t in tokenlist if (type(t["id"]) == int) and (t["head"] != 0)])
                assert nx.is_tree(G)
                trees.append(G)
                root = [t["id"] for t in tokenlist if t["head"] == 0]
                assert len(root) == 1
                roots.append(root[0])
                if len(sents) == 100:
                    break
        return trees, roots, sents, lengths

    def plot(self, id):
        """Convenience tree plotter, isn't really used any time though"""
        nx.draw(self.trees[id], pos=nx.spring_layout(self.trees[id]), with_labels=True, font_weight='bold')

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        return self.Sentence(tokens=self.sents[index], 
                             length=self.lengths[index], 
                             tree=self.trees[index],
                             embeddings=self.read_embeddings(index))
        
    def wordpiece_tokenize(self):
        """Subword spans are cached in order to perform averaging of subword embeddings of a token"""
        subtokens = []
        spans = []
        for sent in tqdm(self.sents, desc="[wordpiece tokenizing]"):
            sentence_spans = []
            sentence_subtokens = []
            for token in sent:
                token_subtokens = self.tokenizer.wordpiece_tokenizer.tokenize(token)
                sentence_spans.append((len(sentence_subtokens), len(sentence_subtokens) + len(token_subtokens)))
                sentence_subtokens += token_subtokens
            subtokens.append(sentence_subtokens)
            spans.append(sentence_spans)
        return subtokens, spans

    def decay(self, embeddings, index):
        """Weights decay exponentially as distance between tokens increases"""
        decay = self.length_to_decay[ self.lengths[index] ]
        dec = torch.matmul(decay, torch.tensor(embeddings, device=self.args["device"]).double())
        return dec / self.length_to_norm[ self.lengths[index] ]

    def compute_decays_and_norms(self):
        """Computes and caches weight decay and normalization for all sentence lengths in dataset"""
        decays = {}
        norms = {}
        for length in tqdm(set(self.lengths), desc="[computing decay]"):
            range_ = torch.arange(length, device=self.args["device"])
            distances = [torch.arange(0-i, length-i, device=self.args["device"]) for i in range_]
            abs_distances_to_neighbours = torch.vstack(distances).abs().double()
            # fp precision gotcha strikes if you use .pow without double precision
            squared_abs_distances_to_neighbours = torch.pow(2, abs_distances_to_neighbours)
            decays.update({ length: torch.true_divide(1, squared_abs_distances_to_neighbours) })

            norm = 1 / torch.pow(2, range_.double())
            norm = torch.tensor([norm[:i+1].sum() for i in range_], device=self.args["device"])
            norms.update({length: (norm + norm.flip([0])).unsqueeze(1)})
        return decays, norms

    def custom_pad(self, batch):
        """Pads sequences with zero and labels with -1."""
        seqs = [torch.tensor(x[0].embeddings, device=self.args["device"]) for x in batch]
        lengths = torch.tensor([x[0].length for x in batch], device=self.args["device"])
        padded_seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        label_shape = batch[0][1].shape
        max_len = int(lengths.max())
        labels_new_shape = [max_len for _ in label_shape]

        padded_labels = [-torch.ones(*labels_new_shape, device=self.args["device"]) for _ in seqs]
        for i, length in enumerate(lengths):
            if len(label_shape) == 1:
                padded_labels[i][:length] = batch[i][1]
            elif len(label_shape) == 2:
                padded_labels[i][:length, :length] = batch[i][1]
            else:
                raise ValueError("Labels must be either 1D or 2D.")
        padded_labels = torch.stack(padded_labels)
        return padded_seqs, padded_labels, lengths, batch
    
    def loader(self):
        return torch.utils.data.DataLoader(self, batch_size=self.batch_size, collate_fn=self.custom_pad)

class OneWordDataset(ProbingDataset):
    """Computes Depth labels for probing task"""
    def __init__(self, args, path_to_conllu, cached_labels=False):
        super().__init__(args, path_to_conllu, cached_labels)
        self.labels = self.cached_labels if self.cached_labels else self.compute_labels()

    def __getitem__(self, index):
        return super().__getitem__(index), self.labels[index]

    def compute_labels(self):
        """Computes depth labels using precomputed trees"""
        depths = []
        for tree, root_id, sent in tqdm(zip(self.trees, self.roots, self.sents), desc="[computing depth labels]"):
            sent_depths = torch.zeros(len(sent))
            for node_id in range(len(sent)):
                # all node ids must be > 0 because of conllu notation
                sent_depths[node_id] = nx.shortest_path_length(tree, root_id, node_id+1)
            depths.append(sent_depths)
        return depths
