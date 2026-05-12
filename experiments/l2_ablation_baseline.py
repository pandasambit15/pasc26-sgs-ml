#!/usr/bin/env python3
"""
Experiment 1: L2 Regularization Ablation Study (FINAL VERSION)
===============================================================

Systematically tests different L2 regularization strengths to find
optimal values for each architecture (MLP, ResMLP, TabTransformer).

INTEGRATIONS:
- ✅ Uses your actual model architectures
- ✅ Uses your CoefficientDataset with proper scalers
- ✅ Keeps Huber loss for regression
- ✅ Includes positivity constraint
- ✅ Ready for production use
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
    create_model_from_architecture,
    predictions_dict_from_model_output,
    compute_regression_metrics
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class L2RegularizationExperiment:
    """
    Comprehensive L2 regularization ablation study.
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info(f"Using device: {self.device}")
        
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
            
            # Forward pass
            model_output = model(features)
            predictions = predictions_dict_from_model_output(model_output)
            
            # Compute loss with Huber + positivity
            total_loss, losses = compute_multitask_loss_with_huber(
                predictions, targets, task_weights
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate losses
            epoch_losses['total'] += total_loss.item()
            for k, v in losses.items():
                if k in epoch_losses:
                    epoch_losses[k] += v.item()
            n_batches += 1
        
        # Average losses
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
                
                # Forward pass
                model_output = model(features)
                predictions = predictions_dict_from_model_output(model_output)
                
                # Compute loss
                total_loss, losses = compute_multitask_loss_with_huber(
                    predictions, targets, task_weights
                )
                
                # Accumulate losses
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
                
                # Collect predictions and targets
                for key in all_preds.keys():
                    if key == 'regime':
                        all_preds[key].append(
                            predictions[key].argmax(dim=1).cpu()
                        )
                        all_targets[key].append(targets[key].cpu())
                    else:
                        all_preds[key].append(predictions[key].cpu())
                        all_targets[key].append(targets[key].cpu())
        
        # Concatenate all batches
        for key in all_preds.keys():
            all_preds[key] = torch.cat(all_preds[key]).numpy()
            all_targets[key] = torch.cat(all_targets[key]).numpy()
        
        # Compute metrics
        metrics = {}
        
        # Regression metrics
        for key in ['viscosity', 'diffusivity', 'richardson']:
            reg_metrics = compute_regression_metrics(
                all_preds[key], all_targets[key]
            )
            for metric_name, value in reg_metrics.items():
                metrics[f'{key}_{metric_name}'] = value
        
        # Classification accuracy
        regime_acc = (
            all_preds['regime'] == all_targets['regime']
        ).mean()
        metrics['regime_accuracy'] = regime_acc
        
        # Parameter L2 norm
        param_norm = sum(
            p.pow(2).sum().item() for p in model.parameters()
        )
        metrics['param_l2_norm'] = np.sqrt(param_norm)
        
        return metrics
    
    def train_single_config(self, architecture: str, lambda_val: float,
                           seed: int, train_loader: DataLoader,
                           val_loader: DataLoader, 
                           test_loader: DataLoader) -> Dict:
        """
        Train a single model configuration.
        
        Returns:
            Dictionary with training history and final metrics
        """
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create model
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {architecture} with λ={lambda_val}, seed={seed}")
        logger.info(f"{'='*60}")
        
        model = create_model_from_architecture(architecture).to(self.device)
        
        # Optimizer with L2 regularization (weight_decay)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=lambda_val  # This is L2 regularization!
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
        
        # Task weights (equal weights for L2 study)
        task_weights = self.config['training']['task_weights']
        
        # Training loop
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
            # Train
            train_losses = self.train_one_epoch(
                model, train_loader, optimizer, task_weights
            )
            
            # Validate
            val_losses = self.evaluate(model, val_loader, task_weights)
            
            # Update scheduler
            scheduler.step(val_losses['total'])
            
            # Record history
            history['train_loss'].append(train_losses['total'])
            history['val_loss'].append(val_losses['total'])
            history['epoch'].append(epoch)
            
            # Early stopping
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
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Compute final test metrics
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
        
        # Load data
        logger.info("Loading data...")
        train_data, val_data, test_data = load_experiment_data(
            data_dir, scaler_dir
        )
        
        # Create data loaders - USE FLAT CONFIG
        batch_size = self.config['batch_size']
        num_workers = self.config['num_workers']
    
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, 
            pin_memory=True
        )
        test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, 
            pin_memory=True
        )

        # Results storage
        all_results = []
        
        # Iterate over all configurations
        total_experiments = (
            len(self.config['architectures']) * 
            len(self.config['lambda_values']) * 
            len(self.config['random_seeds'])
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {total_experiments} experiments...")
        logger.info(f"{'='*60}")
        
        exp_counter = 0
        for arch in self.config['architectures']:
            for lambda_val in self.config['lambda_values']:
                for seed in self.config['random_seeds']:
                    exp_counter += 1
                    logger.info(
                        f"\n[{exp_counter}/{total_experiments}] "
                        f"{arch}, λ={lambda_val}, seed={seed}"
                    )
                    
                    # Train model
                    results = self.train_single_config(
                        arch, lambda_val, seed,
                        train_loader, val_loader, test_loader
                    )
                    
                    # Store results
                    result_entry = {
                        'architecture': arch,
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
                    
                    # Save intermediate results
                    df = pd.DataFrame(all_results)
                    df.to_csv(
                        output_dir / 'l2_results_interim.csv', 
                        index=False
                    )
        
        # Save final results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_dir / 'l2_results_final.csv', index=False)
        
        # Generate analysis and plots
        self._analyze_results(results_df, output_dir)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment complete! Results saved to {output_dir}")
        logger.info(f"{'='*60}")
        
        return results_df
    
    def _analyze_results(self, results_df: pd.DataFrame, output_dir: Path):
        """Generate comprehensive analysis and visualizations."""
        
        # Aggregate across random seeds
        agg_results = results_df.groupby(['architecture', 'lambda']).agg({
            'viscosity_r2': ['mean', 'std'],
            'diffusivity_r2': ['mean', 'std'],
            'richardson_r2': ['mean', 'std'],
            'regime_accuracy': ['mean', 'std'],
            'param_l2_norm': ['mean', 'std'],
            'train_val_gap': ['mean', 'std']
        }).reset_index()
        
        agg_results.to_csv(
            output_dir / 'l2_results_aggregated.csv', 
            index=False
        )
        
        # Generate plots
        self._plot_performance_curves(results_df, output_dir)
        self._plot_overfitting_analysis(results_df, output_dir)
        self._plot_parameter_norm(results_df, output_dir)
        self._generate_summary_table(results_df, output_dir)
    
    def _plot_performance_curves(self, df: pd.DataFrame, output_dir: Path):
        """Plot performance metrics vs lambda."""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = [
            'viscosity_r2', 'diffusivity_r2', 
            'richardson_r2', 'regime_accuracy'
        ]
        titles = [
            'Viscosity R²', 'Diffusivity R²', 
            'Richardson R²', 'Regime Accuracy'
        ]
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            for arch in df['architecture'].unique():
                arch_data = df[df['architecture'] == arch]
                grouped = arch_data.groupby('lambda')[metric].agg(
                    ['mean', 'std']
                )
                
                ax.errorbar(
                    grouped.index, grouped['mean'], yerr=grouped['std'],
                    marker='o', label=arch, capsize=5, 
                    linewidth=2, markersize=8
                )
            
            ax.set_xlabel('L2 Regularization (λ)', fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(f'{title} vs Regularization Strength', fontsize=14)
            ax.set_xscale('log')
            ax.grid(alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(
            output_dir / 'l2_performance_curves.png', 
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        logger.info("✓ Saved performance curves")
    
    def _plot_overfitting_analysis(self, df: pd.DataFrame, output_dir: Path):
        """Plot train-validation gap."""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for arch in df['architecture'].unique():
            arch_data = df[df['architecture'] == arch]
            grouped = arch_data.groupby('lambda')['train_val_gap'].agg(
                ['mean', 'std']
            )
            
            ax.errorbar(
                grouped.index, grouped['mean'], yerr=grouped['std'],
                marker='o', label=arch, capsize=5, 
                linewidth=2, markersize=8
            )
        
        ax.axhline(y=0, color='red', linestyle='--', 
                   label='No gap (perfect fit)')
        ax.set_xlabel('L2 Regularization (λ)', fontsize=12)
        ax.set_ylabel('Train-Val Loss Gap', fontsize=12)
        ax.set_title('Overfitting Analysis', fontsize=14)
        ax.set_xscale('log')
        ax.grid(alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(
            output_dir / 'l2_overfitting_analysis.png', 
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        logger.info("✓ Saved overfitting analysis")
    
    def _plot_parameter_norm(self, df: pd.DataFrame, output_dir: Path):
        """Plot parameter L2 norm vs lambda."""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for arch in df['architecture'].unique():
            arch_data = df[df['architecture'] == arch]
            grouped = arch_data.groupby('lambda')['param_l2_norm'].agg(
                ['mean', 'std']
            )
            
            ax.errorbar(
                grouped.index, grouped['mean'], yerr=grouped['std'],
                marker='o', label=arch, capsize=5, 
                linewidth=2, markersize=8
            )
        
        ax.set_xlabel('L2 Regularization (λ)', fontsize=12)
        ax.set_ylabel('Parameter L2 Norm', fontsize=12)
        ax.set_title('Parameter Magnitude vs Regularization', fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(
            output_dir / 'l2_parameter_norm.png', 
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        logger.info("✓ Saved parameter norm plot")
    
    def _generate_summary_table(self, df: pd.DataFrame, output_dir: Path):
        """Generate summary table of best configurations."""
        
        summary = []
        for arch in df['architecture'].unique():
            arch_data = df[df['architecture'] == arch]
            
            # Find best lambda based on viscosity R²
            best_idx = arch_data.groupby('lambda')['viscosity_r2'].mean().idxmax()
            best_data = arch_data[arch_data['lambda'] == best_idx]
            
            summary.append({
                'Architecture': arch,
                'Best λ': best_idx,
                'Visc R²': (
                    f"{best_data['viscosity_r2'].mean():.3f} ± "
                    f"{best_data['viscosity_r2'].std():.3f}"
                ),
                'Diff R²': (
                    f"{best_data['diffusivity_r2'].mean():.3f} ± "
                    f"{best_data['diffusivity_r2'].std():.3f}"
                ),
                'Ri R²': (
                    f"{best_data['richardson_r2'].mean():.3f} ± "
                    f"{best_data['richardson_r2'].std():.3f}"
                ),
                'Regime Acc': (
                    f"{best_data['regime_accuracy'].mean():.1%} ± "
                    f"{best_data['regime_accuracy'].std():.1%}"
                )
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(output_dir / 'l2_best_configs.csv', index=False)
        
        # LaTeX table
        with open(output_dir / 'l2_best_configs.tex', 'w') as f:
            f.write(summary_df.to_latex(index=False, escape=False))
        
        logger.info("✓ Saved summary tables")

def main():
    parser = argparse.ArgumentParser(
        description='L2 Regularization Experiment'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config YAML file'
    )
    # These override config file if provided
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
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--num-workers', type=int, default=None,
        help='Number of workers (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {args.config}")
        
        # Extract paths from YAML structure
        data_dir = args.data_dir or config['paths']['data_dir']
        scaler_dir = args.scaler_dir or config['paths']['scaler_dir']
        output_dir = args.output_dir or config.get('exp1_output_dir', 'experiments/l2_regularization')
        
        # Extract training params (handle nested structure)
        batch_size = args.batch_size or config['training'].get('batch_size', 8192)
        num_workers = args.num_workers or config['compute'].get('num_workers', 4)
        
        # Build experiment config (flatten structure for internal use)
        exp_config = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'architectures': config.get('architectures', ['MLP', 'ResMLP', 'TabTransformer']),
            'lambda_values': config.get('lambda_values', [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
            'training': config['training'],
            'random_seeds': config.get('random_seeds', [42, 123, 456]),
            'output_dir': output_dir
        }
    else:
        # No config file - require command-line args
        if not args.data_dir or not args.scaler_dir:
            parser.error("--data-dir and --scaler-dir are required when not using --config")
        
        logger.info("No config file provided, using command-line arguments and defaults")
        
        data_dir = args.data_dir
        scaler_dir = args.scaler_dir
        output_dir = args.output_dir or 'experiments/l2_regularization'
        
        exp_config = {
            'batch_size': args.batch_size or 8192,
            'num_workers': args.num_workers or 4,
            'architectures': ['MLP', 'ResMLP', 'TabTransformer'],
            'lambda_values': [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'training': {
                'epochs': 100,
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
    
    # Run experiment
    experiment = L2RegularizationExperiment(exp_config)
    results_df = experiment.run_experiment(
        Path(data_dir), 
        Path(scaler_dir)
    )
    
    logger.info("\n✓ Experiment 1 complete!")


#if __name__ == "__main__":
#    main()

def main_old():
    parser = argparse.ArgumentParser(
        description='L2 Regularization Experiment'
    )
    parser.add_argument(
        '--data-dir', type=str, required=True,
        help='Directory with features.npy, visc_coeff.npy, etc.'
    )
    parser.add_argument(
        '--scaler-dir', type=str, required=True,
        help='Directory with scaler .pkl files'
    )
    parser.add_argument(
        '--output-dir', type=str, 
        default='experiments/l2_regularization',
        help='Output directory for results'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config YAML (optional)'
    )
   
    parser.add_argument(
        '--batch-size', type=int, default=8192,
        help='Batch size for training'
    )
    parser.add_argument(
        '--num-workers', type=int, default=4,
        help='Number of data loading workers'
    )

    args = parser.parse_args()
    
    # Default configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    else:
        config = {
            # Flatten the structure
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'architectures': ['MLP', 'ResMLP', 'TabTransformer'],
            'lambda_values': [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'training': {
                'epochs': 100,
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
            'output_dir': args.output_dir
        }

    # If config from YAML doesn't have output_dir, use args
    if 'output_dir' not in config:  # ← ADD THIS CHECK!
        config['output_dir'] = args.output_dir

    # Ensure batch_size and num_workers are set
    if 'batch_size' not in config:
        config['batch_size'] = args.batch_size
    if 'num_workers' not in config:
        config['num_workers'] = args.num_workers

    # Run experiment
    experiment = L2RegularizationExperiment(config)
    results_df = experiment.run_experiment(
        Path(args.data_dir), 
        Path(args.scaler_dir)
    )
    
    logger.info("\n✓ Experiment 1 complete!")


if __name__ == "__main__":
    main()
