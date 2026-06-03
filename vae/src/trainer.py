import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))

import config 

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        scheduler = None,
        device = config.DEVICE,
        val_loader = None,
        lambda_regular = 0.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.scheduler = scheduler
        self.val_loader = val_loader
        self.device = device
        self.lambda_regular = lambda_regular
    
    def train_epoch(self):
        self.model.train()
        self.model.to(config.DEVICE)
        
        running_loss = 0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(config.DEVICE)
            self.optimizer.zero_grad()
            pred = self.model(data)
            loss = self.criterion(pred, data)
            # L2 regularization on model parameters
            if self.lambda_regular and self.lambda_regular != 0.0:
                l2_reg = torch.tensor(0.0, device=self.device)
                for p in self.model.parameters():
                    l2_reg = l2_reg + torch.sum(p.pow(2))
                loss = loss + self.lambda_regular * l2_reg
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
        train_loss = running_loss / len(self.train_loader)
        return train_loss
        
    def validate(self):
        self.model.eval()
        self.model.to(config.DEVICE)
        running_loss = 0
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation", leave=False)
            for batch_idx, (data, _) in enumerate(progress_bar):
                data = data.to(config.DEVICE)
                pred = self.model(data)
                loss = self.criterion(pred, data)
                # Add L2 regularization to validation loss if enabled
                if self.lambda_regular and self.lambda_regular != 0.0:
                    l2_reg = torch.tensor(0.0, device=self.device)
                    for p in self.model.parameters():
                        l2_reg = l2_reg + torch.sum(p.pow(2))
                    loss = loss + self.lambda_regular * l2_reg

                running_loss += loss.item()
        return running_loss / len(self.val_loader)
        
    def fit(self, num_epochs):
        best_val_loss = float("inf")
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if epoch % 20 == 0 or (epoch + 1) == config.EPOCHS:
                print(
                    f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss: .4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"vae": self.model.state_dict()},
                    config.MODEL_WEIGHTS_PATH,
                )
            
            if self.scheduler is not None:
                self.scheduler.step()

        self.save_training_progress_plot(train_losses, val_losses)

    def save_training_progress_plot(self, train_losses, val_losses):
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Progress")
        plt.legend()
        plt.tight_layout()
        output_path = Path(config.training_progress_dir) / "loss_curve.png"
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()