import torch
import torch.nn as nn
from tqdm.auto import tqdm


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
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.scheduler = scheduler
        self.val_loader = val_loader
        self.device = device
    
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
                
                running_loss += loss.item()
        return running_loss / len(self.val_loader)
        
    def fit(self, num_epochs):
        best_val_loss = float("inf")
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
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