"""Training utilities for Material GNN models"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from typing import Optional, Dict, List
import numpy as np
from tqdm import tqdm


class MaterialGNNTrainer:
    """
    Trainer class for Material GNN models.
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: Optional[torch.device] = None,
                 learning_rate: float = 0.01,
                 weight_decay: float = 5e-4):
        """
        Initialize trainer.
        
        Args:
            model: GNN model to train.
            device: Device to run training on (CPU/GPU).
            learning_rate: Learning rate for optimizer.
            weight_decay: Weight decay for optimizer.
        """
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, data: Data, targets: Optional[torch.Tensor] = None) -> float:
        """
        Train for one epoch.
        
        Args:
            data: PyTorch Geometric Data object.
            targets: Target values for supervised learning (optional).
        
        Returns:
            Average training loss.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        data = data.to(self.device)
        if targets is not None:
            targets = targets.to(self.device)
        
        # Forward pass
        if targets is not None:
            # Supervised learning: property prediction
            predictions = self.model(data.x, data.edge_index)
            loss = self.criterion(predictions, targets)
        else:
            # Unsupervised learning: embedding learning
            embeddings = self.model(data.x, data.edge_index)
            # Simple reconstruction loss (can be customized)
            loss = torch.mean(embeddings ** 2)  # L2 regularization
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, data: Data, targets: Optional[torch.Tensor] = None) -> float:
        """
        Validate model.
        
        Args:
            data: PyTorch Geometric Data object.
            targets: Target values for validation (optional).
        
        Returns:
            Validation loss.
        """
        self.model.eval()
        
        with torch.no_grad():
            data = data.to(self.device)
            if targets is not None:
                targets = targets.to(self.device)
            
            if targets is not None:
                predictions = self.model(data.x, data.edge_index)
                loss = self.criterion(predictions, targets)
            else:
                embeddings = self.model(data.x, data.edge_index)
                loss = torch.mean(embeddings ** 2)
        
        return loss.item()
    
    def train(self,
              train_data: Data,
              num_epochs: int = 100,
              train_targets: Optional[torch.Tensor] = None,
              val_data: Optional[Data] = None,
              val_targets: Optional[torch.Tensor] = None,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train model for multiple epochs.
        
        Args:
            train_data: Training data.
            num_epochs: Number of training epochs.
            train_targets: Training targets (optional).
            val_data: Validation data (optional).
            val_targets: Validation targets (optional).
            verbose: Whether to print training progress.
        
        Returns:
            Dictionary with training and validation losses.
        """
        self.train_losses = []
        self.val_losses = []
        
        for epoch in tqdm(range(num_epochs), desc="Training", disable=not verbose):
            # Training
            train_loss = self.train_epoch(train_data, train_targets)
            self.train_losses.append(train_loss)
            
            # Validation
            if val_data is not None:
                val_loss = self.validate(val_data, val_targets)
                self.val_losses.append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}", end="")
                if val_data is not None:
                    print(f", Val Loss: {val_loss:.4f}")
                else:
                    print()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def get_embeddings(self, data: Data) -> np.ndarray:
        """
        Get material embeddings from trained model.
        
        Args:
            data: PyTorch Geometric Data object.
        
        Returns:
            Material embeddings as numpy array.
        """
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            embeddings = self.model(data.x, data.edge_index)
            return embeddings.cpu().numpy()
    
    def save_model(self, path: str):
        """Save model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

