"""Classes for specifying structural probes."""

import torch.nn as nn
import torch

class Probe(nn.Module):
    pass

class TwoWordProbe(Probe):
    """Computes squared L2 distance after projection by a matrix."""
    def __init__(self, args):
        super().__init__()
        self.rank = args["probe"]["rank"]
        self.hidden_dim = args["featurizer"]["dim"]
        self.proj = nn.Parameter(data=torch.zeros(self.hidden_dim, self.rank))
        nn.init.uniform_(tensor=self.proj, a=-0.05, b=0.05)
        self.to(args["device"])

    def forward(self, x):
        x = torch.matmul(x.float(), self.proj)
        batch_size, seqlen, rank = x.size()
        x = x.unsqueeze(2)
        x = x.expand(-1, -1, seqlen, -1)
        diffs = x - x.transpose(1, 2)
        diffs = diffs.pow(2)
        squared_distances = torch.sum(diffs, -1)
        return squared_distances

class OneWordProbe(Probe):
    """Computes squared L2 norms after projection by a matrix"""
    def __init__(self, args):
        super().__init__()
        self.rank = args["probe"]["rank"]
        self.hidden_dim = args["featurizer"]["dim"]
        self.proj = nn.Parameter(data=torch.zeros(self.hidden_dim, self.rank))
        nn.init.uniform_(tensor=self.proj, a=-0.05, b=0.05)
        self.to(args["device"])

    def forward(self, x):
        x = torch.matmul(x.float(), self.proj)
        batch_size, seqlen, rank = x.size()
        norms = torch.bmm(x.view(batch_size*seqlen, 1, rank), x.view(batch_size*seqlen, rank, 1))
        norms = norms.view(batch_size, seqlen)
        return norms
