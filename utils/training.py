"""Classes for training and running inference on probe"""

from tqdm.auto import tqdm
import torch

def predict(probe, dataset):
    """Predicts distance or depth labels"""
    probe.eval()
    predictions_by_batch = []
    for x, y_true, lengths, observations in tqdm(dataset.loader(), desc="[predicting]"):
        y_pred = probe(x)
        predictions_by_batch.append(y_pred.detach().cpu().numpy())
    predictions = [preds for batch in predictions_by_batch for preds in batch] #comment out to get batched predictions
    return predictions

class ProbeTrainer:
    
    def __init__(self, args):
        self.args = args
        self.max_epochs = args["probe_training"]["max_epochs"]
        self.initial_lr = args["probe_training"]["initial_lr"]
        self.probe_params_path = args["probe"]["params_path"]

    def set_optimizer(self, probe):
        """Sets optimizer and scheduler for training.
        Learning rate decays by a factor of 0.1 if loss doesn't improve after an epoch"""
        self.optimizer = torch.optim.Adam(probe.parameters(), lr=self.initial_lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.1, patience=0)

    def train_until_convergence(self, probe, loss, train_loader, dev_loader):
        """Trains a probe untill loss on dev dataset does not improve
        by more than 0.0001 for 5 epochs straight.

        Writes parameters of th probe to the disk at the location specified by config."""
        self.set_optimizer(probe)
        min_dev_loss = float("inf")
        min_dev_loss_epoch = -1

        for epoch_id in range(self.max_epochs):
            
            epoch_train_loss = 0
            epoch_dev_loss = 0

            epoch_train_epoch_count = 0
            epoch_dev_epoch_count = 0

            epoch_train_loss_count = 0
            epoch_dev_loss_count = 0

            for x, y_true, lengths, observation in tqdm(train_loader, desc=f"[training batch, epoch {epoch_id}]"):
                probe.train()
                self.optimizer.zero_grad()
                y_pred = probe(x)
                batch_loss, batch_total_sents = loss(y_pred, y_true, lengths)
                batch_loss.backward()

                epoch_train_loss += batch_loss.detach().cpu().numpy() * batch_total_sents.detach().cpu().numpy()
                epoch_train_epoch_count += 1
                epoch_train_loss_count += batch_total_sents.detach().cpu().numpy()
                
                self.optimizer.step()
                
            for x, y_true, lengths, observation in tqdm(dev_loader, desc=f"[dev batch, epoch {epoch_id}]"):
                self.optimizer.zero_grad()
                probe.eval()
                y_pred = probe(x)
                batch_loss, batch_total_sents = loss(y_pred, y_true, lengths)
                epoch_dev_loss += batch_loss.detach().cpu().numpy() * batch_total_sents.detach().cpu().numpy()
                epoch_dev_loss_count += batch_total_sents.detach().cpu().numpy()
                epoch_dev_epoch_count += 1

            self.scheduler.step(epoch_dev_loss)
            tqdm.write(f"[epoch {epoch_id}] Mean train loss: {epoch_train_loss/epoch_train_loss_count}. Mean dev loss: {epoch_dev_loss/epoch_dev_loss_count}")
            if (epoch_dev_loss / epoch_dev_loss_count) < (min_dev_loss - 0.0001):
                torch.save(probe.state_dict(), self.probe_params_path)
                min_dev_loss = epoch_dev_loss / epoch_dev_loss_count
                min_dev_loss_epoch = epoch_id
                tqdm.write("Saving probe parameters")

            # early stopping in case last five epochs yielded no result
            elif min_dev_loss_epoch < (epoch_id - 4):
                tqdm.write("Early stopping")
                break
