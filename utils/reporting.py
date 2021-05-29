"""Class for computing and repoting probe evaluation metrics"""

from collections import defaultdict
import os

from tqdm.auto import tqdm
from scipy.stats import spearmanr
import numpy as np
import networkx as nx
from conllu import parse_incr

import yaml
import torch

class Reporter:
    """Reporting class for parse-distance and parse-depth probing tasks, reports on test set"""
    def __init__(self, args, cached_labels):
        self.args = args
        self.split_ = "test"
        self.rank = args["probe"]["rank"]
        self.reporting_root = args["reporting"]["reporting_root"]

        if int(args["featurizer"]["layer"]) < 0:
            self.layer = str(13 + int(args["featurizer"]["layer"]))
        else:
            self.layer = args["featurizer"]["layer"]
            
        self.predictions_path = args["probe"]["predictions_path"]
##        self.labels_path = f'cached_labels/{args["probe"]["task"]}/test.labels'
        self.predictions = torch.load(self.predictions_path, map_location=torch.device('cpu'))
##        self.labels = torch.load(self.labels_path, map_location=torch.device('cpu'))
        self.labels = cached_labels
        self.lengths = [len(label) for label in self.labels]
        self.trees, self.sents, self.uposes = self.trees_sents_uposes()
        self.total_deps, self.correct_deps, self.total_span_len = defaultdict(int), defaultdict(int), defaultdict(int)
        self.deps = self.compute_deps()
        
         self.report_spearmanr()
        if args["probe"]["task"] == "parse-distance":
            self.report_uuas()
         elif args["probe"]["task"] == "parse-depth":
             self.report_root_acc()
        
    def report_spearmanr(self):
        """Computes Spearman correlation between predicted and gold labels."""
        sentlen_to_spearmanrs = defaultdict(list)
        for y_pred, y_true, length in tqdm(zip(self.predictions, self.labels, self.lengths), 
                                           total=len(self.predictions), desc="[computing spearmanr]"):
            if self.args["probe"]["task"] == "parse-distance":
                y_pred = y_pred[:length, :length]
                corr = [spearmanr(pred, gold).correlation for pred, gold in zip(y_pred, y_true)]
            elif args["probe"]["task"] == "parse-depth":
                y_pred = y_pred[:length]
                corr = [spearmanr(y_pred, y_true).correlation]
            sentlen_to_spearmanrs[length].extend(corr)
        sentlen_to_mean_spearmanr = {k: np.mean(v) for k, v in sentlen_to_spearmanrs.items()}    
        mean = np.mean([v for k, v in sentlen_to_mean_spearmanr.items() if k in range(5, 51)])

        with open(self.reporting_root+"/"+f"rank{self.rank}-layer-{self.layer}-{self.split_}.spearmanr", "w") as f:
            for length in sorted(sentlen_to_spearmanrs):
                f.write(f"{length}\t{sentlen_to_spearmanrs[length]}\n")
        with open(self.reporting_root+"/"+f"rank{self.rank}-layer-{self.layer}-{self.split_}.spearmanr_5-50_mean", "w") as f:
            f.write(str(mean))

        print(f"{self.rank} - rank. {self.layer} - layer.\tMean Spearman correlation - {mean}")
    
    def report_uuas(self):
        """Computes UUAS score for a dataset"""
        uspan_total = 0
        uspan_correct = 0
        for y_pred, length, tree, upos, sent, edge2dep in tqdm(zip(self.predictions, self.lengths, self.trees, self.uposes, 
                                                         self.sents, self.deps), total=len(self.sents), desc="[computing uuas]"):
            G_pred = nx.Graph()
            for i in range(length):
                for j in range(i+1, length):
                    G_pred.add_edge(i+1, j+1, weight=y_pred[i, j].item())
            pred_edges = list(nx.minimum_spanning_tree(G_pred, algorithm="prim").edges)
            pred_nonpunc_edges = [tuple(sorted(e)) for e in pred_edges 
                                  if (upos[int(e[0])-1] != "PUNCT") and (upos[int(e[1])-1] != "PUNCT")]
            gold_nonpunc_edges = [tuple(sorted(e)) for e in list(tree.edges) 
                                  if (upos[int(e[0])-1] != "PUNCT") and (upos[int(e[1])-1] != "PUNCT")]
            
            correct_nonpunct_edges = set(gold_nonpunc_edges) & set(pred_nonpunc_edges)
            uspan_correct += len(correct_nonpunct_edges)
            uspan_total += len(gold_nonpunc_edges)
            
            gold_edge_num = len(gold_nonpunc_edges)

            # TODO: clean up
            d = {"sent_uuas": len(set(gold_nonpunc_edges) & set(pred_nonpunc_edges)) / gold_edge_num if gold_edge_num else 0,
                "length": length, "sent":sent, "gold_edge_num": gold_edge_num,
                "latex": self.print_tikz(pred_nonpunc_edges, gold_nonpunc_edges, sent)}
##            ID2LATEX.append(d)   

            for edge in correct_nonpunct_edges:
                self.correct_deps[edge2dep[edge]] += 1
            for edge in gold_nonpunc_edges:
                self.total_deps[edge2dep[edge]] += 1
                self.total_span_len[edge2dep[edge]] += abs(edge[0]-edge[1])
            
        uuas = uspan_correct / uspan_total
        
         with open(self.reporting_root+"/"+f"rank{self.rank}-layer-{self.layer}-{self.split_}.uuas", "w") as f:
             f.write(str(uuas))
         print(f"{self.rank} - rank. {self.layer} - layer.\tUUAS score - {uuas}")
    
    def report_root_acc(self):
        """Computes root prediction accuracy"""
        correct_root_predictions = 0
        for y_pred, y_true, length, upos in tqdm(zip(self.predictions, self.labels, self.lengths, self.uposes), 
                                               total=len(self.predictions), desc="[computing root accuracy]"):
            y_pred = y_pred[:length]
            nonpunc_depth_sorted_indices = sorted([(i, depth) for i, depth in enumerate(y_pred) 
                                                   if upos[i] != "PUNCT"], key=lambda tup: tup[1])
            correct_root_predictions += list(y_true).index(0) == nonpunc_depth_sorted_indices[0][0]
        root_acc = correct_root_predictions / len(self.predictions)
        with open(self.reporting_root+"/"+f"rank{self.rank}-layer-{self.layer}-{self.split_}.root_acc", "w") as f:
            f.write(str(root_acc))
        print(f"{self.rank} - rank. {self.layer} - layer.\tRoot accuracy score - {root_acc}")
        
    def trees_sents_uposes(self):
        """Builds sentence trees"""
        trees = []
        sents = []
        uposes = []
        path = self.args["corpus"]["corpus_root"]+"/"+self.args["corpus"]["test_path"]
        with open(path, encoding="utf-8") as data_file:
            for tokenlist in tqdm(parse_incr(data_file), total=len(self.predictions), desc="[building trees]"):
                upos = [t["upos"] for t in tokenlist if type(t["id"]) == int]
                uposes.append(upos)
                
                sents.append([t["form"] for t in tokenlist if type(t["id"]) == int])
                G = nx.Graph()
                # TODO: remove trees, leave only edges
                G.add_edges_from([(t["id"], t["head"]) for t in tokenlist 
                                  if (type(t["id"]) == int) and (t["head"] != 0)])
                assert nx.is_tree(G)
                trees.append(G)
        return trees, sents, uposes
    
    def compute_deps(self):
        """Parses nonpunct dependency labels"""
        deps = []
        path = self.args["corpus"]["corpus_root"]+"/"+self.args["corpus"]["test_path"]
        with open(path, encoding="utf-8") as data_file:
            for tokenlist in tqdm(parse_incr(data_file), total=len(self.predictions), desc="[computing deps]"):
                
                d = {}
                for t in tokenlist:
                    if (type(t["id"]) == int) and (t["deps"][0][0] not in {'punct', 'root'}):
                        head_id = t["head"]
                        dep = t["deps"][0][0]
                        d.update({ (head_id, t["id"]): dep, (t["id"], head_id) : dep })
                deps.append(d)
        return deps
    
    def print_tikz(self, predicted_edges, gold_edges, words):
        """Returns LaTeX visualizing gold and predicted dependencies with tikz-dependency LaTeX package"""
        s = """\\begin{dependency}[hide label, edge unit distance=.5ex]
        \\begin{deptext}[column sep=0.05cm]
        """ 
        s += "\\& ".join([x.replace('$', '\$').replace('&', '+') for x in words]) + " \\\\" + '\n'
        s += "\\end{deptext}" + '\n'
        for i, j in gold_edges:
            s += '\\depedge{{{}}}{{{}}}{{{}}}\n'.format(i, j, '.')
        for i, j in predicted_edges:
            s += '\\depedge[edge style={{red!60!}}, edge below]{{{}}}{{{}}}{{{}}}\n'.format(i, j, '.')
        s += '\\end{dependency}\n'
        return s