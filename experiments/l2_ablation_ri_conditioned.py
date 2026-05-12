#!/usr/bin/env python3
"""
L2 Regularization Experiment for Richardson-Conditioned Models
==============================================================

IDENTICAL to l2_regularization_experiment_final.py but uses Ri-conditioned models.
This enables direct comparison: same data, same training, only difference is Ri conditioning.

Usage:
    python l2_regularization_ri_conditioned.py --config experiments_config.yaml --architecture MLP
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
from tqdm import tqdm
import logging
import argparse

from experiment_utils import (
    load_experiment_data,
    compute_multitask_loss_with_huber,
    create_ri_conditioned_model,  # NEW: Use Ri-conditioned models
    predictions_dict_from_model_output,
    compute_regression_metrics
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class L2RegularizationExperimentRiConditioned:
    """
    L2 regularization ablation for Richardson-conditioned models.
    Code is IDENTICAL to L2RegularizationExperiment except model creation.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        logger.info("🔬 Using Richardson-conditioned models")
        
    def train_one_epoch(self, model: nn.Module, train_loader: DataLoader,
                       optimizer: optim.Optimizer, 
                       task_weights: Dict) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        epoch_losses = {
            'total': 0.0, 'viscosity': 0.0, 'diffusivity': 0.0,
            'richardson': 0.0, 'regime': 0.0, 'positivity': 0.0
        }
        n_batches = 0
        
        for batch in train_loader:
            features, targets = batch
            features = features.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            model_output = model(features)
            predictions = predictions_dict_from_model_output(model_output)
            
            total_loss, losses = compute_multitask_loss_with_huber(
                predictions, targets, task_weights
            )
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses['total'] += total_loss.item()
            for k, v in losses.items():
                if k in epoch_losses:
                    epoch_losses[k] += v.item()
            n_batches += 1
        
        return {k: v / n_batches for k, v in epoch_losses.items()}
    
    def evaluate(self, model: nn.Module, val_loader: DataLoader,
                task_weights: Dict) -> Dict[str, float]:
        """Evaluate model on validation set."""
        model.eval()
        epoch_losses = {
            'total': 0.0, 'viscosity': 0.0, 'diffusivity': 0.0,
            'richardson': 0.0, 'regime': 0.0, 'positivity': 0.0
        }
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features, targets = batch
                features = features.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                model_output = model(features)
                predictions = predictions_dict_from_model_output(model_output)
                
                total_loss, losses = compute_multitask_loss_with_huber(
                    predictions, targets, task_weights
                )
                
                epoch_losses['total'] += total_loss.item()
                for k, v in losses.items():
                    if k in epoch_losses:
                        epoch_losses[k] += v.item()
                n_batches += 1
        
        return {k: v / n_batches for k, v in epoch_losses.items()}
    
    def compute_metrics(self, model: nn.Module, 
                       test_loader: DataLoader) -> Dict:
        """Compute comprehensive test metrics."""
        model.eval()
        
        all_preds = {
            'viscosity': [], 'diffusivity': [], 
            'richardson': [], 'regime': []
        }
        all_targets = {
            'viscosity': [], 'diffusivity': [], 
            'richardson': [], 'regime': []
        }
        
        with torch.no_grad():
            for batch in test_loader:
                features, targets = batch
                features = features.to(self.device)
                
                model_output = model(features)
                predictions = predictions_dict_from_model_output(model_output)
                
                for key in all_preds.keys():
                    if key == 'regime':
                        all_preds[key].append(
                            predictions[key].argmax(dim=1).cpu()
                        )
                        all_targets[key].append(targets[key].cpu())
                    else:
                        all_preds[key].append(predictions[key].cpu())
                        all_targets[key].append(targets[key].cpu())
        
        for key in all_preds.keys():
            all_preds[key] = torch.cat(all_preds[key]).numpy()
            all_targets[key] = torch.cat(all_targets[key]).numpy()
        
        metrics = {}
        
        for key in ['viscosity', 'diffusivity', 'richardson']:
            reg_metrics = compute_regression_metrics(
                all_preds[key], all_targets[key]
            )
            for metric_name, value in reg_metrics.items():
                metrics[f'{key}_{metric_name}'] = value
        
        regime_acc = (
            all_preds['regime'] == all_targets['regime']
        ).mean()
        metrics['regime_accuracy'] = regime_acc
        
        param_norm = sum(
            p.pow(2).sum().item() for p in model.parameters()
        )
        metrics['param_l2_norm'] = np.sqrt(param_norm)
        
        return metrics
    
    def train_single_config(self, architecture: str, lambda_val: float,
                           seed: int, train_loader: DataLoader,
                           val_loader: DataLoader, 
                           test_loader: DataLoader) -> Dict:
        """Train a single model configuration."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Ri-conditioned {architecture} with λ={lambda_val}, seed={seed}")
        logger.info(f"{'='*60}")
        
        # CRITICAL DIFFERENCE: Use Ri-conditioned model
        model = create_ri_conditioned_model(architecture).to(self.device)
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=lambda_val
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
        
        task_weights = self.config['training']['task_weights']
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        epochs = self.config['training']['epochs']
        patience = self.config['training']['early_stopping_patience']
        
        for epoch in range(epochs):
            train_losses = self.train_one_epoch(
                model, train_loader, optimizer, task_weights
            )
            
            val_losses = self.evaluate(model, val_loader, task_weights)
            
            scheduler.step(val_losses['total'])
            
            history['train_loss'].append(train_losses['total'])
            history['val_loss'].append(val_losses['total'])
            history['epoch'].append(epoch)
            
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                patience_counter = 0
                best_model_state = {
                    k: v.cpu().clone() 
                    for k, v in model.state_dict().items()
                }
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"Train Loss = {train_losses['total']:.6f}, "
                    f"Val Loss = {val_losses['total']:.6f}"
                )
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        model.load_state_dict(best_model_state)
        
        test_metrics = self.compute_metrics(model, test_loader)
        
        logger.info(f"Test R² - Visc: {test_metrics['viscosity_r2']:.4f}, "
                   f"Diff: {test_metrics['diffusivity_r2']:.4f}, "
                   f"Ri: {test_metrics['richardson_r2']:.4f}")
        
        return {
            'history': history,
            'test_metrics': test_metrics,
            'best_epoch': len(history['epoch']) - patience_counter,
            'best_val_loss': best_val_loss
        }
    
    def run_experiment(self, data_dir: Path, scaler_dir: Path):
        """Run complete L2 regularization experiment."""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Loading data...")
        train_data, val_data, test_data = load_experiment_data(
            data_dir, scaler_dir
        )
        
        batch_size = self.config['batch_size']
        num_workers = self.config['num_workers']
    
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        all_results = []
        
        total_experiments = (
            len(self.config['architectures']) * 
            len(self.config['lambda_values']) * 
            len(self.config['random_seeds'])
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {total_experiments} Ri-conditioned experiments...")
        logger.info(f"{'='*60}")
        
        exp_counter = 0
        for arch in self.config['architectures']:
            for lambda_val in self.config['lambda_values']:
                for seed in self.config['random_seeds']:
                    exp_counter += 1
                    logger.info(
                        f"\n[{exp_counter}/{total_experiments}] "
                        f"Ri-{arch}, λ={lambda_val}, seed={seed}"
                    )
                    
                    results = self.train_single_config(
                        arch, lambda_val, seed,
                        train_loader, val_loader, test_loader
                    )
                    
                    result_entry = {
                        'architecture': f'Ri-{arch}',  # Mark as Ri-conditioned
                        'lambda': lambda_val,
                        'seed': seed,
                        **results['test_metrics'],
                        'best_epoch': results['best_epoch'],
                        'best_val_loss': results['best_val_loss'],
                        'final_train_loss': results['history']['train_loss'][-1],
                        'final_val_loss': results['history']['val_loss'][-1],
                        'train_val_gap': (
                            results['history']['train_loss'][-1] - 
                            results['history']['val_loss'][-1]
                        )
                    }
                    all_results.append(result_entry)
                    
                    df = pd.DataFrame(all_results)
                    df.to_csv(
                        output_dir / 'l2_results_ri_interim.csv', 
                        index=False
                    )
        
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_dir / 'l2_results_ri_final.csv', index=False)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Ri-conditioned experiment complete! Results saved to {output_dir}")
        logger.info(f"{'='*60}")
        
        return results_df


def main():
    parser = argparse.ArgumentParser(
        description='L2 Regularization Experiment for Ri-Conditioned Models'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--architecture', type=str, required=True,
        choices=['MLP', 'ResMLP', 'TabTransformer'],
        help='Architecture to test'
    )
    parser.add_argument(
        '--data-dir', type=str, default=None,
        help='Directory with features.npy, visc_coeff.npy, etc.'
    )
    parser.add_argument(
        '--scaler-dir', type=str, default=None,
        help='Directory with scaler .pkl files'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {args.config}")
        
        data_dir = args.data_dir or config['paths']['data_dir']
        scaler_dir = args.scaler_dir or config['paths']['scaler_dir']
        output_dir = args.output_dir or f"experiments/l2_regularization_ri/{args.architecture}"
        
        batch_size = config['training'].get('batch_size', 8192)
        num_workers = config['compute'].get('num_workers', 4)
        
        exp_config = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'architectures': [args.architecture],  # Single architecture
            'lambda_values': config.get('lambda_values', [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
            'training': config['training'],
            'random_seeds': config.get('random_seeds', [42, 123, 456]),
            'output_dir': output_dir
        }
    else:
        if not args.data_dir or not args.scaler_dir:
            parser.error("--data-dir and --scaler-dir are required when not using --config")
        
        data_dir = args.data_dir
        scaler_dir = args.scaler_dir
        output_dir = args.output_dir or f'experiments/l2_regularization_ri/{args.architecture}'
        
        exp_config = {
            'batch_size': 8192,
            'num_workers': 4,
            'architectures': [args.architecture],
            'lambda_values': [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'training': {
                'epochs': 300,
                'learning_rate': 1e-3,
                'early_stopping_patience': 15,
                'task_weights': {
                    'viscosity': 1.0,
                    'diffusivity': 1.0,
                    'richardson': 1.0,
                    'regime': 1.0,
                    'positivity': 1.0
                }
            },
            'random_seeds': [42, 123, 456],
            'output_dir': output_dir
        }
    
    experiment = L2RegularizationExperimentRiConditioned(exp_config)
    results_df = experiment.run_experiment(
        Path(data_dir), 
        Path(scaler_dir)
    )
    
    logger.info("\n✅ Ri-conditioned L2 experiment complete!")


if __name__ == "__main__":
    main()
