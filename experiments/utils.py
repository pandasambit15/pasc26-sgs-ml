#!/usr/bin/env python3
"""
Shared Utilities for PASC 2026 Experiments
==========================================

Common functions and classes used across all experiments.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Tuple
import joblib
import logging

logger = logging.getLogger(__name__)


class DictFormatDataset(Dataset):
    """
    Wraps CoefficientDataset to convert tuple output to dict format
    required by experiment scripts.
    """
    
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get from base dataset: (features, visc, diff, ri, regime)
        features, visc, diff, ri, regime = self.base_dataset[self.indices[idx]]
        
        # Convert to dict format for experiments
        targets = {
            'viscosity': visc.squeeze(),
            'diffusivity': diff.squeeze(),
            'richardson': ri.squeeze(),
            'regime': regime
        }
        
        return features, targets

def load_experiment_data(data_dir: Path, scaler_dir: Path, 
                        train_split: float = 0.7, 
                        val_split: float = 0.1,
                        seed: int = 42):
    """
    Load data for experiments using your existing CoefficientDataset.
    """
    # Import here to avoid circular dependencies
    try:
        from train_new_coeff import CoefficientDataset
    except ImportError:
        import sys
        parent = Path(__file__).parent.parent
        sys.path.insert(0, str(parent))
        from train_new_coeff import CoefficientDataset
    
    logger.info(f"Loading data from {data_dir}")
    full_dataset = CoefficientDataset(data_dir)
    
    # Create train/val/test splits
    indices = np.arange(len(full_dataset))
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    n_total = len(full_dataset)
    n_train = int(train_split * n_total)
    n_val = int(val_split * n_total)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    logger.info(f"Dataset splits - Train: {len(train_indices)}, "
                f"Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # FIXED: Check if setup_scalers accepts scaler_dir parameter
    import inspect
    sig = inspect.signature(full_dataset.setup_scalers)
    params = list(sig.parameters.keys())
    
    if len(params) == 2 and 'scaler_dir' in params:
        # New version: accepts scaler_dir
        full_dataset.setup_scalers(train_indices, scaler_dir)
    elif len(params) == 1:
        # Old version: only accepts train_indices
        # Copy scalers to data_dir if they're in a different location
        if scaler_dir != data_dir:
            logger.warning(
                f"Your CoefficientDataset.setup_scalers() doesn't accept scaler_dir parameter.\n"
                f"Make sure scaler .pkl files are in {data_dir}"
            )
        full_dataset.setup_scalers(train_indices)
    else:
        raise ValueError(f"Unexpected setup_scalers signature: {sig}")
    
    # Wrap datasets to convert to dict format
    train_data = DictFormatDataset(full_dataset, train_indices)
    val_data = DictFormatDataset(full_dataset, val_indices)
    test_data = DictFormatDataset(full_dataset, test_indices)
    
    return train_data, val_data, test_data

def load_experiment_data_old(data_dir: Path, scaler_dir: Path, 
                        train_split: float = 0.7, 
                        val_split: float = 0.1,
                        seed: int = 42):
    """
    Load data for experiments using your existing CoefficientDataset.
    
    Args:
        data_dir: Directory with features.npy, visc_coeff.npy, etc.
        scaler_dir: Directory with scaler .pkl files
        train_split: Fraction for training (default 0.7)
        val_split: Fraction for validation (default 0.1)
        seed: Random seed for reproducibility
        
    Returns:
        train_data, val_data, test_data (wrapped datasets)
    """
    # Import here to avoid circular dependencies
    from train_new_coeff import CoefficientDataset
    
    logger.info(f"Loading data from {data_dir}")
    full_dataset = CoefficientDataset(data_dir)
    
    # Create train/val/test splits
    indices = np.arange(len(full_dataset))
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    n_total = len(full_dataset)
    n_train = int(train_split * n_total)
    n_val = int(val_split * n_total)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    logger.info(f"Dataset splits - Train: {len(train_indices)}, "
                f"Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # Setup scalers (must be pre-computed or will be created from training data)
    full_dataset.setup_scalers(train_indices, scaler_dir)
    
    # Wrap datasets to convert to dict format
    train_data = DictFormatDataset(full_dataset, train_indices)
    val_data = DictFormatDataset(full_dataset, val_indices)
    test_data = DictFormatDataset(full_dataset, test_indices)
    
    return train_data, val_data, test_data


def compute_multitask_loss_with_huber(predictions: Dict, targets: Dict,
                                      task_weights: Dict,
                                      delta: float = 1.0) -> Tuple[torch.Tensor, Dict]:
    """
    Compute multi-task loss using Huber loss for regression tasks
    and includes positivity constraint.
    
    Args:
        predictions: Model predictions dict with keys 
                    ['viscosity', 'diffusivity', 'richardson', 'regime']
        targets: Ground truth dict with same keys
        task_weights: Dictionary of task weights
        delta: Huber loss delta parameter
        
    Returns:
        total_loss, losses_dict
    """
    losses = {}
    
    # Regression tasks: Use Huber loss (robust to outliers)
    losses['viscosity'] = F.huber_loss(
        predictions['viscosity'], 
        targets['viscosity'],
        delta=delta
    )
    
    losses['diffusivity'] = F.huber_loss(
        predictions['diffusivity'], 
        targets['diffusivity'],
        delta=delta
    )
    
    losses['richardson'] = F.huber_loss(
        predictions['richardson'], 
        targets['richardson'],
        delta=delta
    )
    
    # Classification task: Cross-entropy
    losses['regime'] = F.cross_entropy(
        predictions['regime'], 
        targets['regime']
    )
    
    # Positivity constraint: Penalize negative predictions for coefficients
    losses['positivity'] = (
        F.softplus(-predictions['viscosity']).mean() + 
        F.softplus(-predictions['diffusivity']).mean()
    )
    
    # Weighted total loss
    total_loss = sum(
        task_weights.get(k, 0) * losses[k] 
        for k in losses.keys()
    )
    
    return total_loss, losses


def create_model_from_architecture(architecture: str, n_features: int = 54):
    """
    Factory function to create models from architecture name.
    
    Args:
        architecture: 'MLP', 'ResMLP', or 'TabTransformer'
        n_features: Number of input features
        
    Returns:
        Model instance
    """
    if architecture == 'MLP':
        from train_new_coeff import UnifiedSGSCoefficientNetwork
        return UnifiedSGSCoefficientNetwork(n_features=n_features)
    
    elif architecture == 'ResMLP':
        from train_resmlp import ResMLPNetwork
        return ResMLPNetwork(n_features=n_features)
    
    elif architecture == 'TabTransformer':
        from train_tab_transformer import TabTransformerNetwork
        return TabTransformerNetwork(n_features=n_features)
    
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def predictions_dict_from_model_output(model_output: Tuple) -> Dict:
    """
    Convert model tuple output to dict format for experiments.
    
    Your models return: (visc, diff, ri, regime)
    Experiments expect: {'viscosity': ..., 'diffusivity': ..., etc.}
    
    Args:
        model_output: Tuple of (visc, diff, richardson, regime)
        
    Returns:
        Dictionary with proper keys
    """
    visc, diff, ri, regime = model_output
    
    return {
        'viscosity': visc.squeeze(-1) if visc.dim() > 1 else visc,
        'diffusivity': diff.squeeze(-1) if diff.dim() > 1 else diff,
        'richardson': ri.squeeze(-1) if ri.dim() > 1 else ri,
        'regime': regime
    }


def compute_regression_metrics(predictions: np.ndarray, 
                               targets: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics.
    
    Returns:
        Dictionary with r2, rmse, mae, correlation
    """
    # R²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else -np.inf
    
    # RMSE
    rmse = np.sqrt(np.mean((targets - predictions) ** 2))
    
    # MAE
    mae = np.mean(np.abs(targets - predictions))
    
    # Correlation
    if len(predictions) > 1:
        corr = np.corrcoef(targets, predictions)[0, 1]
    else:
        corr = 0.0
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'correlation': corr
    }

def create_ri_conditioned_model(architecture: str, n_features: int = 54):
    """
    Factory function to create Richardson-conditioned models.
    
    Args:
        architecture: 'MLP', 'ResMLP', or 'TabTransformer'
        n_features: Number of input features
        
    Returns:
        Ri-conditioned model instance
    """
    if architecture == 'MLP':
        from multitask_neural_network_v2 import RiConditionedMLP
        return RiConditionedMLP(n_features=n_features)
    
    elif architecture == 'ResMLP':
        from multitask_neural_network_v2 import RiConditionedResMLP
        return RiConditionedResMLP(n_features=n_features)
    
    elif architecture == 'TabTransformer':
        from multitask_neural_network_v2 import RiConditionedTabTransformer
        return RiConditionedTabTransformer(n_features=n_features)
    
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
