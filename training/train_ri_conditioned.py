#!/usr/bin/env python3
"""
Simple Training Script for Richardson-Conditioned Models
========================================================

Usage:
    python train_ri_conditioned_simple.py \
        --data-dir /path/to/data \
        --scaler-dir /path/to/scalers \
        --architecture MLP \
        --checkpoint-dir ./checkpoints_ri_mlp
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
import argparse
from collections import defaultdict
from tqdm import tqdm

# Import dataset from existing scripts
from train_resmlp import CoefficientDataset

# Import Ri-conditioned models
from multitask_neural_network_v2 import (
    RiConditionedMLP,
    RiConditionedResMLP,
    RiConditionedTabTransformer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleTrainer:
    """Simple trainer for Ri-conditioned models."""
    
    def __init__(self, model, config: dict, checkpoint_dir: str):
        self.model = model
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        self.best_val_loss = float('inf')
        self.history = defaultdict(list)

    def compute_loss(self, preds, targets):
        """Compute multi-task loss."""
        pred_visc, pred_diff, pred_ri, pred_regime = preds
        true_visc, true_diff, true_ri, true_regime = targets

        # Huber loss for regression
        loss_visc = F.huber_loss(pred_visc, true_visc)
        loss_diff = F.huber_loss(pred_diff, true_diff)
        loss_ri = F.huber_loss(pred_ri, true_ri)
        
        # Cross-entropy for classification
        loss_regime = F.cross_entropy(pred_regime, true_regime)
        
        # Positivity constraint
        loss_pos = F.softplus(-pred_visc).mean() + F.softplus(-pred_diff).mean()
        
        total_loss = (
            self.config['lambda_visc'] * loss_visc +
            self.config['lambda_diff'] * loss_diff +
            self.config['lambda_ri'] * loss_ri +
            self.config['lambda_regime'] * loss_regime +
            self.config['lambda_pos'] * loss_pos
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'visc': loss_visc.item(),
            'diff': loss_diff.item(),
            'ri': loss_ri.item(),
            'regime': loss_regime.item(),
            'positivity': loss_pos.item()
        }
        
        return total_loss, loss_dict

    def train_epoch(self, data_loader):
        """Train for one epoch."""
        self.model.train()
        epoch_losses = defaultdict(list)
        
        for batch in tqdm(data_loader, desc="Training"):
            targets = [x.to(self.device) for x in batch]
            features = targets.pop(0)
            
            self.optimizer.zero_grad()
            preds = self.model(features)
            loss, loss_dict = self.compute_loss(preds, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            for key, value in loss_dict.items():
                epoch_losses[key].append(value)
        
        return {key: np.mean(values) for key, values in epoch_losses.items()}

    def validate(self, data_loader):
        """Validate."""
        self.model.eval()
        epoch_losses = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Validation", leave=False):
                targets = [x.to(self.device) for x in batch]
                features = targets.pop(0)
                
                preds = self.model(features)
                loss, loss_dict = self.compute_loss(preds, targets)
                
                for key, value in loss_dict.items():
                    epoch_losses[key].append(value)
        
        return {key: np.mean(values) for key, values in epoch_losses.items()}

    def train(self, train_loader, val_loader, num_epochs):
        """Full training loop."""
        early_stopping_patience = self.config.get('early_stopping_patience', 15)
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_losses = self.train_epoch(train_loader)
            val_losses = self.validate(val_loader)
            
            self.scheduler.step(val_losses['total'])
            
            # Log
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_losses['total']:.6f} | "
                f"Val Loss: {val_losses['total']:.6f}"
            )
            
            # Track history
            for key, value in train_losses.items():
                self.history[f'train_{key}'].append(value)
            for key, value in val_losses.items():
                self.history[f'val_{key}'].append(value)
            
            # Early stopping
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
                patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                logger.info(f"  ✓ New best model (val_loss: {self.best_val_loss:.6f})")
            else:
                patience_counter += 1
                logger.info(f"  No improvement ({patience_counter}/{early_stopping_patience})")
            
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered!")
                break
        
        logger.info(f"\n✅ Training complete! Best val loss: {self.best_val_loss:.6f}")

    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint."""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': dict(self.history)
        }
        
        torch.save(state, self.checkpoint_dir / 'latest_checkpoint.pt')
        
        if is_best:
            torch.save(state, self.checkpoint_dir / 'best_checkpoint.pt')


def main():
    parser = argparse.ArgumentParser(description='Train Ri-conditioned model')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--scaler-dir', type=str, required=True)
    parser.add_argument('--architecture', type=str, required=True,
                       choices=['MLP', 'ResMLP', 'TabTransformer'])
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_ri')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    
    args = parser.parse_args()
    
    config = {
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'early_stopping_patience': 30,
        'lambda_visc': 1.0,
        'lambda_diff': 1.0,
        'lambda_ri': 1.0,
        'lambda_regime': 1.0,
        'lambda_pos': 1.0
    }
    
    # Load dataset
    full_dataset = CoefficientDataset(Path(args.data_dir))
    
    indices = np.arange(len(full_dataset))
    np.random.shuffle(indices)
    split_idx = int(0.9 * len(full_dataset))
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    
    full_dataset.setup_scalers(train_indices, scaler_dir=Path(args.scaler_dir))
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(
        full_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        full_dataset, batch_size=args.batch_size,
        sampler=val_sampler, num_workers=4, pin_memory=True
    )
    
    # Create Ri-conditioned model
    if args.architecture == 'MLP':
        model = RiConditionedMLP()
    elif args.architecture == 'ResMLP':
        model = RiConditionedResMLP()
    else:
        model = RiConditionedTabTransformer()
    
    logger.info(f"Created Ri-conditioned {args.architecture}")
    
    # Train
    trainer = SimpleTrainer(model, config, checkpoint_dir=args.checkpoint_dir)
    trainer.train(train_loader, val_loader, args.epochs)


if __name__ == '__main__':
    main()
