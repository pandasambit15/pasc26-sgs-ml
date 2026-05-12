#!/usr/bin/env python3
"""
Final Training Script for Unified SGS Coefficient Network (with History)
=======================================================================

This script trains the multi-task network and incorporates robust, standard
training practices.

Key Features:
- ✅ **Uses Existing Scalers**: Loads .pkl scaler files from the data processor.
- ✅ **Fallback Mechanism**: Creates RobustScalers if .pkl files are not found.
- ✅ **Training History**: Tracks and saves all loss components for plotting.
- ✅ **Early Stopping**: Halts training if validation loss does not improve.
- ✅ **LR Scheduler, Gradient Clipping, Dropout**: Standard best practices.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
import logging
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import sys
import argparse
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== MODEL ARCHITECTURE ====================

class UnifiedSGSCoefficientNetwork(nn.Module):
    """Multi-task network for SGS coefficient prediction."""
    
    def __init__(self, n_features=54):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3)
        )
        self.visc_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.diff_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.richardson_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.regime_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))
    
    def forward(self, x):
        features = self.backbone(x)
        return self.visc_head(features), self.diff_head(features), self.richardson_head(features), self.regime_head(features)


# ==================== DATASET ====================

class CoefficientDataset(Dataset):
    """Loads data and handles scaling logic."""
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        
        # Load data arrays
        self.features = np.load(self.data_dir / 'features.npy')
        self.visc_coeff = np.load(self.data_dir / 'visc_coeff.npy').reshape(-1, 1)
        self.diff_coeff = np.load(self.data_dir / 'diff_coeff.npy').reshape(-1, 1)
        self.ri_smag = np.load(self.data_dir / 'richardson.npy').reshape(-1, 1)
        self.regime = np.load(self.data_dir / 'regime.npy')
        
    def setup_scalers(self, train_indices):
        """Fit scalers on the training split or load them."""
        scaler_files = ['feature_scaler.pkl', 'visc_scaler.pkl', 'diff_scaler.pkl', 'richardson_scaler.pkl']
        scaler_paths = [self.data_dir / f for f in scaler_files]

        if all(p.exists() for p in scaler_paths):
            logger.info("Loading existing scalers from data directory.")
            self.feature_scaler = joblib.load(scaler_paths[0])
            self.visc_scaler = joblib.load(scaler_paths[1])
            self.diff_scaler = joblib.load(scaler_paths[2])
            self.ri_scaler = joblib.load(scaler_paths[3])
        else:
            logger.warning("Scaler .pkl files not found! Creating new RobustScalers from training split and saving them.")
            self.feature_scaler = RobustScaler().fit(self.features[train_indices])
            self.visc_scaler = RobustScaler().fit(self.visc_coeff[train_indices])
            self.diff_scaler = RobustScaler().fit(self.diff_coeff[train_indices])
            self.ri_scaler = RobustScaler().fit(self.ri_smag[train_indices])
            
            joblib.dump(self.feature_scaler, scaler_paths[0])
            joblib.dump(self.visc_scaler, scaler_paths[1])
            joblib.dump(self.diff_scaler, scaler_paths[2])
            joblib.dump(self.ri_scaler, scaler_paths[3])
            
        # Apply scaling to the entire dataset
        self.features = self.feature_scaler.transform(self.features)
        self.visc_coeff = self.visc_scaler.transform(self.visc_coeff)
        self.diff_coeff = self.diff_scaler.transform(self.diff_coeff)
        self.ri_smag = self.ri_scaler.transform(self.ri_smag)
        logger.info("Dataset scaled successfully.")

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (torch.from_numpy(self.features[idx]).float(), torch.from_numpy(self.visc_coeff[idx]).float(),
                torch.from_numpy(self.diff_coeff[idx]).float(), torch.from_numpy(self.ri_smag[idx]).float(),
                torch.tensor(self.regime[idx]).long())


# ==================== TRAINER ====================

class CoefficientTrainer:
    """Trainer with history tracking, early stopping, and other best practices."""
    
    def __init__(self, model, config: Dict, checkpoint_dir: str):
        self.model = model
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.early_stopping_counter = 0
        self.loss_weights = {k: config[k] for k in ['lambda_visc', 'lambda_diff', 'lambda_ri', 'lambda_regime', 'lambda_pos']}
        
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.history = defaultdict(list)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Saves model checkpoint with history."""
        state = {'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'scheduler_state_dict': self.scheduler.state_dict(),
                 'best_val_loss': self.best_val_loss, 'history': dict(self.history)}
        torch.save(state, self.checkpoint_dir / 'latest_checkpoint.pt')
        if is_best:
            torch.save(state, self.checkpoint_dir / 'best_checkpoint.pt')
            logger.info(f"  ✓ Saved best model checkpoint (val_loss: {self.best_val_loss:.6f})")

    def load_checkpoint(self):
        """Loads a checkpoint to resume training."""
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        if not checkpoint_path.exists():
            logger.info("No checkpoint found, starting from scratch.")
            return
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = defaultdict(list, checkpoint.get('history', {}))
        logger.info(f"✓ Resumed from epoch {self.start_epoch}.")

    def compute_loss(self, preds, targets):
        """Computes the multi-task loss with Huber Loss for regression."""
        pred_visc, pred_diff, pred_ri, pred_regime = preds
        true_visc, true_diff, true_ri, true_regime = targets

        loss_visc = F.huber_loss(pred_visc, true_visc)
        loss_diff = F.huber_loss(pred_diff, true_diff)
        loss_ri = F.huber_loss(pred_ri, true_ri)
        loss_regime = F.cross_entropy(pred_regime, true_regime)
        loss_pos = F.softplus(-pred_visc).mean() + F.softplus(-pred_diff).mean()
        
        total_loss = (self.loss_weights['lambda_visc'] * loss_visc + self.loss_weights['lambda_diff'] * loss_diff +
                      self.loss_weights['lambda_ri'] * loss_ri + self.loss_weights['lambda_regime'] * loss_regime +
                      self.loss_weights['lambda_pos'] * loss_pos)
        
        loss_dict = {'total': total_loss.item(), 'visc': loss_visc.item(), 'diff': loss_diff.item(), 'ri': loss_ri.item(),
                     'regime': loss_regime.item(), 'positivity': loss_pos.item(),
                     'regime_acc': (pred_regime.argmax(dim=1) == true_regime).float().mean().item()}
        return total_loss, loss_dict

    def _run_epoch(self, data_loader, is_training: bool):
        """Generic function to run a single epoch of training or validation."""
        self.model.train(is_training)
        epoch_losses = defaultdict(list)
        
        desc = "Training" if is_training else "Validation"
        pbar = tqdm(data_loader, desc=desc, leave=False)
        
        for batch in pbar:
            targets = [x.to(self.device) for x in batch]
            features = targets.pop(0)
            
            if is_training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(is_training):
                preds = self.model(features)
                loss, loss_dict = self.compute_loss(preds, targets)
                
                if is_training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

            for key, value in loss_dict.items():
                epoch_losses[key].append(value)
            pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})

        return {key: np.mean(values) for key, values in epoch_losses.items()}

    def train(self, train_loader, val_loader, num_epochs, resume: bool = False):
        """Full training loop with history tracking and early stopping."""
        if resume: self.load_checkpoint()

        for epoch in range(self.start_epoch, num_epochs):
            train_losses = self._run_epoch(train_loader, is_training=True)
            val_losses = self._run_epoch(val_loader, is_training=False)
            self.scheduler.step(val_losses['total'])
            
            # Log and store history
            logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_losses['total']:.6f} | Val Loss: {val_losses['total']:.6f}")
            for key, value in train_losses.items(): self.history[f'train_{key}'].append(value)
            for key, value in val_losses.items(): self.history[f'val_{key}'].append(value)

            # Early stopping and checkpointing
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                logger.info(f"  Val loss did not improve. Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
            
            self.save_checkpoint(epoch, is_best=is_best)

            if self.early_stopping_counter >= self.early_stopping_patience:
                logger.info(f"EARLY STOPPING triggered after {self.early_stopping_patience} epochs with no improvement.")
                break
        
        logger.info(f"\n✓ Training complete! Best validation loss: {self.best_val_loss:.6f}")


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description='Train SGS coefficient prediction network.')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with processed data from fast_unified_processor.py')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Directory for saving checkpoints')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8192, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--early-stopping-patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint.')
    args = parser.parse_args()
    
    config = {
        'learning_rate': args.lr, 'weight_decay': 1e-4, 'early_stopping_patience': args.early_stopping_patience,
        'lambda_visc': 1.0, 'lambda_diff': 1.0, 'lambda_ri': 1.0, 'lambda_regime': 1.0, 'lambda_pos': 0.0,
    }
    
    # Load full dataset
    full_dataset = CoefficientDataset(Path(args.data_dir))
    
    # Split indices for training and validation
    indices = np.arange(len(full_dataset))
    np.random.shuffle(indices)
    split_idx = int(0.9 * len(full_dataset))
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    
    # Setup scalers using only the training data split
    full_dataset.setup_scalers(train_indices)
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(full_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(full_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)
    
    model = UnifiedSGSCoefficientNetwork()
    trainer = CoefficientTrainer(model, config, checkpoint_dir=args.checkpoint_dir)
    trainer.train(train_loader, val_loader, args.epochs, resume=args.resume)

if __name__ == '__main__':
    main()
