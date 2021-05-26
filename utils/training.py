"""Classes for training and running inference on probe"""

from tqdm.auto import tqdm
import torch

class ProbeTrainer:
    
    def __init__(self, args):
        self.args = args
        self.max_epochs = args["probe_training"]["max_epochs"]
        self.initial_lr = args["probe_training"]["initial_lr"]
        self.probe_params_path = args["probe"]["params_path"]

    def set_optimizer(self, probe):
        self.optimizer = torch.optim.Adam(probe.parameters(), lr=self.initial_lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.1, patience=0)

    def predict(self):
        pass

    def train_until_convergence(self, probe, path_to_embeddings, loss, train_loader, dev_loader):
        
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
