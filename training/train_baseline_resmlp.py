#!/usr/bin/env python3
"""
Training Script for a ResNet-style MLP (ResMLP)
================================================

This script trains a multi-task network using a ResNet-style backbone,
which uses residual connections for improved stability and performance in
deeper networks.

- ✅ Drop-in replacement for the original MLP backbone.
- ✅ Includes all standard training practices (early stopping, etc.).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict
import joblib
import logging
from tqdm import tqdm
import argparse
from collections import defaultdict
from sklearn.preprocessing import RobustScaler
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== DATASET (from previous scripts) ====================

class CoefficientDataset(Dataset):
    """Loads data and handles scaling logic."""
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.features = np.load(self.data_dir / 'features.npy')
        self.visc_coeff = np.load(self.data_dir / 'visc_coeff.npy').reshape(-1, 1)
        self.diff_coeff = np.load(self.data_dir / 'diff_coeff.npy').reshape(-1, 1)
        self.ri_smag = np.load(self.data_dir / 'richardson.npy').reshape(-1, 1)
        self.regime = np.load(self.data_dir / 'regime.npy')
        
    def setup_scalers(self, train_indices, scaler_dir: Path):
        """Fit scalers on the training split or load them from the specified directory."""
        scaler_files = ['feature_scaler.pkl', 'visc_scaler.pkl', 'diff_scaler.pkl', 'richardson_scaler.pkl']
        scaler_paths = [scaler_dir / f for f in scaler_files]

        if all(p.exists() for p in scaler_paths):
            logger.info(f"Loading existing scalers from {scaler_dir}.")
            self.feature_scaler = joblib.load(scaler_paths[0])
            self.visc_scaler = joblib.load(scaler_paths[1])
            self.diff_scaler = joblib.load(scaler_paths[2])
            self.ri_scaler = joblib.load(scaler_paths[3])
        else:
            logger.error(f"Scaler files not found in {scaler_dir}! Please ensure the path is correct.")
            sys.exit(1)
            
        self.features = self.feature_scaler.transform(self.features)
        self.visc_coeff = self.visc_scaler.transform(self.visc_coeff)
        self.diff_coeff = self.diff_scaler.transform(self.diff_coeff)
        self.ri_smag = self.ri_scaler.transform(self.ri_smag)
        logger.info("Dataset scaled successfully.")

    def __len__(self): return len(self.features)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.features[idx]).float(), torch.from_numpy(self.visc_coeff[idx]).float(),
                torch.from_numpy(self.diff_coeff[idx]).float(), torch.from_numpy(self.ri_smag[idx]).float(),
                torch.tensor(self.regime[idx]).long())

# ==================== MODEL ARCHITECTURE ====================

class ResidualBlock(nn.Module):
    """A simple residual block for an MLP."""
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(size, size),
        )
        self.norm = nn.LayerNorm(size)

    def forward(self, x):
        # Apply layers and add the residual connection
        return self.norm(x + self.layers(x))

class ResMLPNetwork(nn.Module):
    """Multi-task network with a ResNet-style backbone."""
    def __init__(self, n_features=54, embed_size=256, num_blocks=4, dropout=0.3):
        super().__init__()
        
        # ResNet-style backbone
        self.backbone = nn.Sequential(
            nn.Linear(n_features, embed_size),
            nn.ReLU(),
            *[ResidualBlock(embed_size, dropout) for _ in range(num_blocks)]
        )
        
        # Prediction heads remain the same
        self.visc_head = nn.Sequential(nn.Linear(embed_size, 128), nn.ReLU(), nn.Linear(128, 1))
        self.diff_head = nn.Sequential(nn.Linear(embed_size, 128), nn.ReLU(), nn.Linear(128, 1))
        self.richardson_head = nn.Sequential(nn.Linear(embed_size, 128), nn.ReLU(), nn.Linear(128, 1))
        self.regime_head = nn.Sequential(nn.Linear(embed_size, 128), nn.ReLU(), nn.Linear(128, 3))

    def forward(self, x):
        features = self.backbone(x)
        return self.visc_head(features), self.diff_head(features), self.richardson_head(features), self.regime_head(features)

# ==================== TRAINER (from train_final_with_history.py) ====================
# The trainer class is identical and can be reused directly.
from train_new_coeff import CoefficientTrainer

# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description='Train a ResMLP for SGS coefficient prediction.')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with processed .npy data files.')
    parser.add_argument('--scaler-dir', type=str, required=True, help='Directory with the .pkl scaler files.')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_resmlp', help='Directory for saving checkpoints.')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--early-stopping-patience', type=int, default=15)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    
    config = {
        'learning_rate': args.lr, 'weight_decay': 1e-5, 
        'early_stopping_patience': args.early_stopping_patience,
        'lambda_visc': 1.0, 'lambda_diff': 1.0, 'lambda_ri': 1.0,
        'lambda_regime': 1.0, 'lambda_pos': 0.0,
    }
    
    full_dataset = CoefficientDataset(Path(args.data_dir))
    
    indices = np.arange(len(full_dataset))
    np.random.shuffle(indices)
    split_idx = int(0.9 * len(full_dataset))
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    
    full_dataset.setup_scalers(train_indices, scaler_dir=Path(args.scaler_dir))
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(full_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(full_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)
    
    model = ResMLPNetwork()
    trainer = CoefficientTrainer(model, config, checkpoint_dir=args.checkpoint_dir)
    trainer.train(train_loader, val_loader, args.epochs, resume=args.resume)

if __name__ == '__main__':
    main()
