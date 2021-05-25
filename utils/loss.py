"""Custom loss classes for structural probing tasks"""

import torch
import torch.nn as nn

class L1DistanceLoss(nn.Module):
    """Custom L1 loss for parse-distance probing task"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dims_to_sum_along = (1, 2)

    def forward(self, y_pred, y_true, lengths):
        """Computes L1 loss on distance matrices, ignores -1 label padding.

        Normalizes first within a sentence by dividing by the square of the sentence length,
        and then accross the batch.

        Args:
            :y_pred: batch of predicted distances
            :y_true: batch of true distances
            :lengths: batch of sentence lengths

        Returns:
            A tuple of:
                :batch_loss: average loss of the batch
                :total_sents: number of sentences in the batch
        """
        mask = (y_true != -1).float()
        y_pred = y_pred * mask
        y_true = y_true * mask
        total_sents = torch.sum((lengths != 0)).float() # there are less than 20 sents in the last batch
        squared_lengths = lengths.pow(2).float()
        if total_sents > 0:
            loss_per_sent = (y_pred - y_true).abs().sum(dim=self.dims_to_sum_along)
            # loss_per_sent = loss_per_sent.sum() / squared_lengths # bug that i initially didn't notice
            loss_per_sent = loss_per_sent / squared_lengths
            batch_loss = loss_per_sent.sum() / total_sents
        else:
            batch_loss = torch.tensor(data=0.0, device=args["device"])
        return batch_loss, total_sents

class L1DepthLoss(nn.Module):
    """Custom L1 loss for parse-depth probing task."""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dims_to_sum_along = 1

    def forward(self, y_pred, y_true, lengths):
        """Computes L1 loss on depth sequences, ignores -1 label padding.

        Normalizes first within a sentence by dividing by the sentence length,
        and then accross the batch.

        Args:
            :y_pred: batch of predicted depths
            :y_true: batch of true depths
            :lengths: batch of sentence lengths

        Returns:
            A tuple of:
                :batch_loss: average loss of the batch
                :total_sents: number of sentences in the batch
        """
        mask = (y_true != -1).float()
        y_pred = y_pred * mask
        y_true = y_true * mask
        total_sents = torch.sum((lengths != 0)).float()
        if total_sents > 0:
            loss_per_sent = (y_pred - y_true).abs().sum(dim=self.dims_to_sum_along)
            loss_per_sent = loss_per_sent / lengths.float()
            batch_loss = loss_per_sent.sum() / total_sents
        else:
            batch_loss = torch.tensor(0.0, device=self.args["device"])
        return batch_loss, total_sents
