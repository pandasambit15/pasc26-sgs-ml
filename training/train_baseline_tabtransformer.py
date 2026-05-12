#!/usr/bin/env python3
"""
Training Script for a TabTransformer-style Network
===================================================

This script trains a multi-task network using a TabTransformer-style
backbone. It learns embeddings for each feature and uses self-attention
to model their interactions.

- ✅ Powerful architecture for finding complex feature interactions.
- ✅ Includes all standard training practices.
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
# The CoefficientDataset class is identical to the one in the ResMLP script.
from train_resmlp import CoefficientDataset

# ==================== MODEL ARCHITECTURE ====================

class TransformerBlock(nn.Module):
    """A single Transformer block with self-attention and a feed-forward network."""
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class TabTransformerNetwork(nn.Module):
    """Multi-task network with a TabTransformer-style backbone."""
    def __init__(self, n_features=54, embed_dim=32, num_layers=4, num_heads=8, ff_dim=128, dropout=0.2):
        super().__init__()
        self.n_features = n_features
        self.embed_dim = embed_dim

        # Learn an embedding for each of the 54 continuous features
        self.feature_embedder = nn.Linear(1, embed_dim)

        # A stack of Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        
        # Flatten the output of the transformer
        final_embedding_size = n_features * embed_dim
        
        # Prediction heads
        self.visc_head = nn.Sequential(nn.Linear(final_embedding_size, 128), nn.ReLU(), nn.Linear(128, 1))
        self.diff_head = nn.Sequential(nn.Linear(final_embedding_size, 128), nn.ReLU(), nn.Linear(128, 1))
        self.richardson_head = nn.Sequential(nn.Linear(final_embedding_size, 128), nn.ReLU(), nn.Linear(128, 1))
        self.regime_head = nn.Sequential(nn.Linear(final_embedding_size, 128), nn.ReLU(), nn.Linear(128, 3))

    def forward(self, x):
        # x has shape (batch_size, 54)
        # Reshape to (batch_size, 54, 1) to "embed" each feature individually
        x = x.unsqueeze(-1)
        
        # Get embeddings: (batch_size, 54, embed_dim)
        embeddings = self.feature_embedder(x)

        # Pass through Transformer layers
        for block in self.transformer_blocks:
            embeddings = block(embeddings)
        
        # Flatten the output for the prediction heads
        # (batch_size, 54 * embed_dim)
        features = embeddings.view(-1, self.n_features * self.embed_dim)
        
        return self.visc_head(features), self.diff_head(features), self.richardson_head(features), self.regime_head(features)

# ==================== TRAINER (from train_final_with_history.py) ====================
from train_new_coeff import CoefficientTrainer

# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description='Train a TabTransformer for SGS coefficient prediction.')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with processed .npy data files.')
    parser.add_argument('--scaler-dir', type=str, required=True, help='Directory with the .pkl scaler files.')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_tabtransformer', help='Directory for saving checkpoints.')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=4096) # Smaller batch size might be needed
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--early-stopping-patience', type=int, default=20) # More patience can be good for transformers
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
    
    model = TabTransformerNetwork()
    trainer = CoefficientTrainer(model, config, checkpoint_dir=args.checkpoint_dir)
    trainer.train(train_loader, val_loader, args.epochs, resume=args.resume)

if __name__ == '__main__':
    main()
