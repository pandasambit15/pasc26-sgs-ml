#!/usr/bin/env python3
"""
Complete Models Analysis with Ri-Conditioning Support - FULL VERSION
====================================================================

Includes ALL visualizations from original script:
- Scatter plots (improved hexbin)
- Distributions (histogram + KDE)
- Vertical profiles
- Heightwise metrics
- Time series evolution
- Physics correlations
- Spatial diagnostics

Usage:
    python run_models_comparison_with_ri_complete.py \
        --mode single \
        --nc-file data.nc \
        --baseline-mlp mlp.pth --ri-mlp ri_mlp.pth \
        --scaler-dir scalers/ --output results/
"""

import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from pathlib import Path
import joblib
import json
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import re
from typing import Dict, Optional, List, Tuple
from multiprocessing import Pool, cpu_count
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from matplotlib.colors import LogNorm, PowerNorm
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# Import BASELINE architectures
from train_new_coeff import UnifiedSGSCoefficientNetwork
from train_resmlp import ResMLPNetwork
from train_tab_transformer import TabTransformerNetwork

# Import RI-CONDITIONED architectures
from multitask_neural_network_v2 import (
    RiConditionedMLP,
    RiConditionedResMLP,
    RiConditionedTabTransformer
)

# Import utilities
from run_best_models_analysis import (
    FastDataLoader,
    FastFeatureExtractor,
    extract_chunk_worker,
    extract_truth_from_netcdf,
    calculate_3d_metrics,
    calculate_heightwise_metrics,
    diagnose_spatial_patterns,
    extract_physical_quantities,
    NumpyEncoder,
    calculate_nonzero_metrics,
    create_nonzero_comparison_table,
    plot_nonzero_comparison,
    plot_zero_fraction_analysis
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="talk")


# ===================================================================
#  UNIFIED INFERENCE ENGINE
# ===================================================================

class UnifiedInferenceEngine:
    """Inference engine supporting both baseline and Ri-conditioned models."""
    
    def __init__(self, 
                 baseline_paths: Optional[Dict[str, Path]] = None,
                 ri_paths: Optional[Dict[str, Path]] = None,
                 scaler_dir: Path = None,
                 n_workers: Optional[int] = None):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_workers = n_workers or max(1, cpu_count() - 2)
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Workers: {self.n_workers}")
        
        self.models = {}
        
        # Load baseline models
        if baseline_paths:
            logger.info("\nLoading BASELINE models (54 features)...")
            baseline_architectures = {
                'MLP': UnifiedSGSCoefficientNetwork,
                'ResMLP': ResMLPNetwork,
                'TabTransformer': TabTransformerNetwork
            }
            
            for name, path in baseline_paths.items():
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                model = baseline_architectures[name](n_features=54)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                model.eval()
                self.models[f'Baseline-{name}'] = model
                logger.info(f"  ✓ Loaded Baseline-{name}")
        
        # Load Ri-conditioned models
        if ri_paths:
            logger.info("\nLoading RI-CONDITIONED models (54 features)...")
            ri_architectures = {
                'MLP': RiConditionedMLP,
                'ResMLP': RiConditionedResMLP,
                'TabTransformer': RiConditionedTabTransformer
            }
            
            for name, path in ri_paths.items():
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                model = ri_architectures[name](n_features=54)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                model.eval()
                self.models[f'Ri-{name}'] = model
                logger.info(f"  ✓ Loaded Ri-{name}")
        
        # Load scalers
        logger.info(f"\nLoading scalers from {scaler_dir}...")
        self.feature_scaler = joblib.load(scaler_dir / 'feature_scaler.pkl')
        self.visc_scaler = joblib.load(scaler_dir / 'visc_scaler.pkl')
        self.diff_scaler = joblib.load(scaler_dir / 'diff_scaler.pkl')
        self.ri_scaler = joblib.load(scaler_dir / 'richardson_scaler.pkl')
        logger.info("  ✓ Scalers loaded")
        
        logger.info(f"\n✅ Engine ready with {len(self.models)} models")
    
    def predict_3d_domain(self, nc_file: Path, time_idx: int, 
                         k_min: int, k_max: int) -> Dict:
        """Run inference for all loaded models."""
        
        # Load data
        data_loader = FastDataLoader(nc_file, time_idx)
        k_max_zn = min(k_max, data_loader.nz_zn - 1)
        
        # Generate coordinates
        coords = [
            (i, j, k) 
            for i in range(data_loader.nx) 
            for j in range(data_loader.ny) 
            for k in range(k_min, k_max_zn + 1)
        ]
        
        logger.info(f"  Processing {len(coords):,} grid points...")
        
        # Parallel feature extraction
        chunk_size = max(1000, len(coords) // (self.n_workers * 4))
        chunks = [coords[i:i + chunk_size] for i in range(0, len(coords), chunk_size)]
        tasks = [(data_loader, chunk) for chunk in chunks]

        all_features = []
        with Pool(self.n_workers) as pool:
            results = list(tqdm(
                pool.imap(extract_chunk_worker, tasks), 
                total=len(tasks), 
                desc="  Extracting features",
                leave=False
            ))
        
        for res in results: 
            all_features.extend([item for item in res if item is not None])
        
        logger.info(f"  Extracted {len(all_features):,} feature vectors")
        
        # Normalize features
        features_scaled = self.feature_scaler.transform(np.array(all_features))
        
        # Run inference for each model
        multi_model_predictions = {}
        
        for model_name, model in self.models.items():
            logger.info(f"  Running {model_name} inference...")
            preds = {'visc': [], 'diff': [], 'ri': [], 'regime': []}
            
            with torch.no_grad():
                for i in range(0, len(features_scaled), 8192):
                    batch = torch.from_numpy(features_scaled[i:i+8192]).float().to(self.device)
                    p_visc, p_diff, p_ri, p_regime = model(batch)
                    preds['visc'].append(p_visc.cpu().numpy())
                    preds['diff'].append(p_diff.cpu().numpy())
                    preds['ri'].append(p_ri.cpu().numpy())
                    preds['regime'].append(p_regime.argmax(dim=1).cpu().numpy())

            # Reshape to 3D
            nx, ny, nk = data_loader.nx, data_loader.ny, k_max_zn - k_min + 1
            
            # Denormalize
            visc_denorm = self.visc_scaler.inverse_transform(np.concatenate(preds['visc']))
            diff_denorm = self.diff_scaler.inverse_transform(np.concatenate(preds['diff']))
            ri_denorm = self.ri_scaler.inverse_transform(np.concatenate(preds['ri']))
            
            multi_model_predictions[model_name] = {
                'visc_coeff': visc_denorm.reshape(nx, ny, nk),
                'diff_coeff': diff_denorm.reshape(nx, ny, nk),
                'richardson': ri_denorm.reshape(nx, ny, nk),
                'regime': np.concatenate(preds['regime']).reshape(nx, ny, nk),
            }
            
            logger.info(f"    Visc: [{visc_denorm.min():.2f}, {visc_denorm.max():.2f}]")
            logger.info(f"    Diff: [{diff_denorm.min():.2f}, {diff_denorm.max():.2f}]")
        
        multi_model_predictions['shared'] = {
            'heights': data_loader.heights[k_min:k_max_zn+1], 
            'k_min': k_min, 
            'k_max': k_max_zn
        }
        
        return multi_model_predictions


# ===================================================================
#  ENHANCED VISUALIZATIONS - IMPROVED SCATTER PLOTS
# ===================================================================

def plot_scatter_comparison_improved(aggregated_preds: Dict, aggregated_truth: Dict,
                                    output_dir: Path, dataset_name: str = "aggregate"):
    """
    IMPROVED scatter plots with better hexbin visualization.
    
    Fixes:
    - Adaptive vmin/vmax based on actual density distribution
    - Better normalization for extreme point counts
    - Optional subsampling for clearer patterns
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Organize models
    model_names = sorted(aggregated_preds.keys())
    n_models = len(model_names)
    
    # Color schemes
    baseline_colors = {'Baseline-MLP': 'Blues', 'Baseline-ResMLP': 'Greens', 
                      'Baseline-TabTransformer': 'Purples'}
    ri_colors = {'Ri-MLP': 'Reds', 'Ri-ResMLP': 'Oranges', 
                'Ri-TabTransformer': 'YlOrRd'}
    cmaps = {**baseline_colors, **ri_colors}
    
    for var_key in ['visc_coeff', 'diff_coeff']:
        logger.info(f"  Creating improved scatter: {var_key}...")
        
        # Layout
        ncols = min(3, n_models)
        nrows = (n_models + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if nrows > 1 else axes
        
        var_label = var_key.replace('_coeff', ' Coefficient').title()
        
        for ax, model_name in zip(axes, model_names):
            pred_flat = aggregated_preds[model_name][var_key]
            true_flat = aggregated_truth[var_key]
            
            # Align lengths
            min_len = min(len(pred_flat), len(true_flat))
            pred_flat = pred_flat[:min_len]
            true_flat = true_flat[:min_len]
            
            # Filter valid + remove extreme outliers (>99.9th percentile)
            valid = ~(np.isnan(pred_flat) | np.isnan(true_flat) | 
                     np.isinf(pred_flat) | np.isinf(true_flat))
            
            if np.any(valid):
                # Remove extreme outliers
                p99_pred = np.percentile(np.abs(pred_flat[valid]), 99.9)
                p99_true = np.percentile(np.abs(true_flat[valid]), 99.9)
                outlier_filter = (np.abs(pred_flat) < p99_pred) & (np.abs(true_flat) < p99_true)
                valid = valid & outlier_filter
            
            pred_valid = pred_flat[valid]
            true_valid = true_flat[valid]
            
            if len(pred_valid) == 0:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                continue
            
            # Subsample if too many points (for better visualization)
            MAX_POINTS_DISPLAY = 1_000_000
            if len(pred_valid) > MAX_POINTS_DISPLAY:
                logger.info(f"    Subsampling {len(pred_valid):,} → {MAX_POINTS_DISPLAY:,} for visualization")
                indices = np.random.choice(len(pred_valid), MAX_POINTS_DISPLAY, replace=False)
                pred_display = pred_valid[indices]
                true_display = true_valid[indices]
            else:
                pred_display = pred_valid
                true_display = true_valid
            
            # Plot range from percentiles (not extreme outliers)
            data_min = np.percentile(true_display, 0.5)
            data_max = np.percentile(true_display, 99.5)
            
            # Hexbin with ADAPTIVE normalization
            cmap = cmaps.get(model_name, 'viridis')
            
            # Create hexbin
            hexbin = ax.hexbin(true_display, pred_display, 
                             gridsize=50,  # Smaller gridsize for better resolution
                             cmap=cmap,
                             mincnt=1,
                             alpha=0.9,
                             edgecolors='none',
                             linewidths=0.2)
            
            # Get actual counts for better normalization
            counts = hexbin.get_array()
            if len(counts) > 0:
                vmin = max(1, np.percentile(counts, 5))   # Ignore lowest 5%
                vmax = np.percentile(counts, 95)           # Saturate at 95th percentile
                
                # Apply PowerNorm for better contrast in dense regions
                hexbin.set_norm(PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax))
            
            # Colorbar
            cbar = plt.colorbar(hexbin, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Point Density', fontsize=9, fontweight='bold')
            
            # 1:1 line
            ax.plot([data_min, data_max], [data_min, data_max], 
                   'k--', linewidth=2.5, alpha=0.7, label='1:1', zorder=10)
            
            # Calculate metrics on FULL dataset (not subsampled)
            r2 = r2_score(true_valid, pred_valid)
            rmse = np.sqrt(mean_squared_error(true_valid, pred_valid))
            
            # Display name
            display_name = model_name.replace('Baseline-', 'B-').replace('Ri-', 'Ri-')
            
            # Stats box
            stats_text = f'N = {len(pred_valid):,}\nR² = {r2:.4f}\nRMSE = {rmse:.4f}'
            ax.text(0.03, 0.97, stats_text,
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
                   family='monospace', fontweight='bold')
            
            ax.set_xlabel('MONC Truth', fontsize=11, fontweight='bold')
            ax.set_ylabel('Model Prediction', fontsize=11, fontweight='bold')
            ax.set_title(display_name, fontsize=12, fontweight='bold')
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim([data_min, data_max])
            ax.set_ylim([data_min, data_max])
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle(f'{var_label} - Model Comparison', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        for ext in ['png', 'pdf']:
            output_file = output_dir / f'scatter_{var_key}_{dataset_name}.{ext}'
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"    ✓ Saved: {output_file.stem}")


# ===================================================================
#  RESTORED VISUALIZATIONS FROM ORIGINAL SCRIPT
# ===================================================================

def plot_distributions(aggregated_preds: Dict, aggregated_truth: Dict,
                      output_dir: Path, dataset_name: str = "aggregate"):
    """Distribution comparison - histogram + KDE."""
    
    output_dir = Path(output_dir)
    
    logger.info(f"\nCreating distribution plots: {dataset_name}")
    
    # Organize models
    baseline_models = sorted([k for k in aggregated_preds.keys() if k.startswith('Baseline-')])
    ri_models = sorted([k for k in aggregated_preds.keys() if k.startswith('Ri-')])
    all_models = baseline_models + ri_models
    
    # Colors
    baseline_colors = {'Baseline-MLP': '#2E86AB', 'Baseline-ResMLP': '#06A77D', 
                      'Baseline-TabTransformer': '#A23B72'}
    ri_colors = {'Ri-MLP': '#E63946', 'Ri-ResMLP': '#F77F00', 
                'Ri-TabTransformer': '#457B9D'}
    colors = {**baseline_colors, **ri_colors}
    
    for var_key in ['visc_coeff', 'diff_coeff']:
        logger.info(f"  {var_key}...")
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Histogram
        ax = axes[0]
        true_flat = aggregated_truth[var_key]
        valid = ~(np.isnan(true_flat) | np.isinf(true_flat))
        true_valid = true_flat[valid]
        
        # Remove extreme outliers for visualization
        p995 = np.percentile(true_valid, 99.5)
        p005 = np.percentile(true_valid, 0.5)
        true_display = true_valid[(true_valid >= p005) & (true_valid <= p995)]
        
        ax.hist(true_display, bins=150, alpha=0.5, color='#C41E3A', 
               label=f'Truth (n={len(true_valid):,})', 
               density=True, edgecolor='none')
        
        for model_name in all_models:
            pred_flat = aggregated_preds[model_name][var_key]
            valid = ~(np.isnan(pred_flat) | np.isinf(pred_flat))
            pred_valid = pred_flat[valid]
            pred_display = pred_valid[(pred_valid >= p005) & (pred_valid <= p995)]
            
            ax.hist(pred_display, bins=150, alpha=0.35, 
                   color=colors.get(model_name, 'gray'),
                   label=f'{model_name} (n={len(pred_valid):,})', 
                   density=True, edgecolor='none')
        
        ax.set_xlabel(var_key.replace('_', ' ').title(), fontsize=13, fontweight='bold')
        ax.set_ylabel('Density', fontsize=13, fontweight='bold')
        ax.set_title('Histogram (0.5-99.5 percentile)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, framealpha=0.95, loc='best')
        ax.grid(True, alpha=0.3)
        
        # KDE
        ax = axes[1]
        
        # Sample for KDE
        kde_sample_size = min(len(true_display), 50000)
        true_sample = np.random.choice(true_display, kde_sample_size, replace=False)
        
        x_range = np.linspace(p005, p995, 500)
        
        try:
            kde_truth = gaussian_kde(true_sample, bw_method='scott')
            ax.plot(x_range, kde_truth(x_range), 
                   color='#C41E3A', linewidth=4, label='Truth', alpha=0.9)
        except:
            logger.warning(f"  Failed to compute KDE for truth")
        
        for model_name in all_models:
            pred_flat = aggregated_preds[model_name][var_key]
            valid = ~(np.isnan(pred_flat) | np.isinf(pred_flat))
            pred_valid = pred_flat[valid]
            pred_display = pred_valid[(pred_valid >= p005) & (pred_valid <= p995)]
            
            if len(pred_display) < 100:
                continue
            
            pred_sample = np.random.choice(pred_display, 
                                          min(len(pred_display), kde_sample_size), 
                                          replace=False)
            
            try:
                kde_pred = gaussian_kde(pred_sample, bw_method='scott')
                ax.plot(x_range, kde_pred(x_range), 
                       color=colors.get(model_name, 'gray'), 
                       linewidth=3, label=model_name, alpha=0.85)
            except:
                logger.warning(f"  Failed to compute KDE for {model_name}")
        
        ax.set_xlabel(var_key.replace('_', ' ').title(), fontsize=13, fontweight='bold')
        ax.set_ylabel('Density', fontsize=13, fontweight='bold')
        ax.set_title('KDE', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, framealpha=0.95, loc='best')
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'{var_key.replace("_", " ").title()} - Distribution Comparison',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        output_file = output_dir / f'distributions_{var_key}_{dataset_name}.png'
        plt.savefig(output_file, dpi=250, bbox_inches='tight')
        plt.close()
        
        logger.info(f"    Saved: {output_file.name}")


def plot_vertical_profiles(predictions: Dict, truth: Dict, output_dir: Path):
    """Enhanced vertical profiles."""
    
    output_dir = Path(output_dir)
    
    logger.info("\nCreating vertical profiles...")
    
    # Organize models
    model_names = sorted([k for k in predictions.keys() if k != 'shared'])
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    
    variables = [
        ('visc_coeff', 'Viscosity Coefficient'), 
        ('diff_coeff', 'Diffusivity Coefficient'), 
        ('richardson', 'Richardson Number')
    ]
    
    # Colors
    baseline_colors = {'Baseline-MLP': '#2E86AB', 'Baseline-ResMLP': '#06A77D', 
                      'Baseline-TabTransformer': '#A23B72'}
    ri_colors = {'Ri-MLP': '#E63946', 'Ri-ResMLP': '#F77F00', 
                'Ri-TabTransformer': '#457B9D'}
    colors = {**baseline_colors, **ri_colors}
    
    heights = predictions['shared']['heights']
    
    for ax, (key, title) in zip(axes, variables):
        # Truth
        true_mean = np.nanmean(truth[key], axis=(0, 1))
        true_std = np.nanstd(truth[key], axis=(0, 1))
        
        ax.plot(true_mean, heights, 'k--', lw=4, 
               label='MONC Truth', zorder=100, alpha=0.9)
        ax.fill_betweenx(heights, true_mean - true_std, true_mean + true_std, 
                        color='gray', alpha=0.2, label='Truth ±1σ', zorder=5)
        
        # Models
        for model_name in model_names:
            preds = predictions[model_name]
            
            pred_mean = np.nanmean(preds[key], axis=(0, 1))
            pred_std = np.nanstd(preds[key], axis=(0, 1))
            
            color = colors.get(model_name, 'blue')
            display_name = model_name.replace('Baseline-', 'B-').replace('Ri-', 'Ri-')
            
            ax.plot(pred_mean, heights, 
                   color=color, lw=2.5, 
                   label=display_name, alpha=0.85, zorder=50)
            ax.fill_betweenx(heights, pred_mean - pred_std, pred_mean + pred_std, 
                            color=color, alpha=0.1, zorder=1)
        
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_ylabel('Height (m)', fontsize=13, fontweight='bold')
        ax.set_xlabel(title, fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'vertical_profiles.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved: {output_file.name}")


def plot_heightwise_metrics(heightwise_metrics: Dict, heights: np.ndarray, 
                            output_dir: Path):
    """Plot height-wise R² and RMSE."""
    
    output_dir = Path(output_dir)
    
    logger.info("\nCreating height-wise metric plots...")
    
    # Colors
    baseline_colors = {'Baseline-MLP': '#2E86AB', 'Baseline-ResMLP': '#06A77D', 
                      'Baseline-TabTransformer': '#A23B72'}
    ri_colors = {'Ri-MLP': '#E63946', 'Ri-ResMLP': '#F77F00', 
                'Ri-TabTransformer': '#457B9D'}
    colors = {**baseline_colors, **ri_colors}
    
    for var_name in ['visc_coeff', 'diff_coeff', 'richardson']:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # R²
        ax = axes[0]
        for model_name, color in colors.items():
            if model_name in heightwise_metrics and var_name in heightwise_metrics[model_name]:
                r2_values = heightwise_metrics[model_name][var_name]['r2']
                display_name = model_name.replace('Baseline-', 'B-').replace('Ri-', 'Ri-')
                ax.plot(r2_values, heights, color=color, linewidth=2.5, 
                       label=display_name, marker='o', markersize=3, alpha=0.85)
        
        ax.axvline(x=0.7, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Target (0.7)')
        ax.set_xlabel('R²', fontsize=13, fontweight='bold')
        ax.set_ylabel('Height (m)', fontsize=13, fontweight='bold')
        ax.set_title(f'{var_name.replace("_", " ").title()} - R² by Height', 
                    fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.5, 1.0])
        
        # RMSE
        ax = axes[1]
        for model_name, color in colors.items():
            if model_name in heightwise_metrics and var_name in heightwise_metrics[model_name]:
                rmse_values = heightwise_metrics[model_name][var_name]['rmse']
                display_name = model_name.replace('Baseline-', 'B-').replace('Ri-', 'Ri-')
                ax.plot(rmse_values, heights, color=color, linewidth=2.5, 
                       label=display_name, marker='o', markersize=3, alpha=0.85)
        
        ax.set_xlabel('RMSE', fontsize=13, fontweight='bold')
        ax.set_ylabel('Height (m)', fontsize=13, fontweight='bold')
        ax.set_title(f'{var_name.replace("_", " ").title()} - RMSE by Height', 
                    fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = output_dir / f'heightwise_{var_name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved: {output_file.name}")


def plot_time_series(series_data: List[Dict], output_dir: Path):
    """Publication-quality time series with timestamps and error bars."""
    
    output_dir = Path(output_dir)
    timestamps = np.array([d['timestamp'] for d in series_data])
    
    # Organize models
    model_names = sorted([k for k in series_data[0]['predictions'].keys()])
    n_models = len(model_names)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Colors
    baseline_colors = {'Baseline-MLP': '#2E86AB', 'Baseline-ResMLP': '#06A77D', 
                      'Baseline-TabTransformer': '#A23B72'}
    ri_colors = {'Ri-MLP': '#E63946', 'Ri-ResMLP': '#F77F00', 
                'Ri-TabTransformer': '#457B9D'}
    colors = {**baseline_colors, **ri_colors, 'Truth': '#000000'}
    
    variables = [
        ('visc_coeff', 'Viscosity Coefficient', axes[0]),
        ('diff_coeff', 'Diffusivity Coefficient', axes[1])
    ]
    
    for var_key, var_label, ax in variables:
        # Truth with error band
        truth_means = np.array([d['truth']['mean'][var_key] for d in series_data])
        truth_stds = np.array([d['truth']['std'][var_key] for d in series_data])
        
        ax.plot(timestamps, truth_means, color='black', linewidth=3, 
               label='MONC Truth', marker='o', markersize=7, zorder=100, linestyle='--')
        ax.fill_between(timestamps, truth_means - truth_stds, truth_means + truth_stds,
                       color='gray', alpha=0.2, label='Truth ±1σ', zorder=50)
        
        # Model predictions with error bars
        for model_name in model_names:
            means = np.array([d['predictions'][model_name]['mean'][var_key] for d in series_data])
            stds = np.array([d['predictions'][model_name]['std'][var_key] for d in series_data])
            
            display_name = model_name.replace('Baseline-', 'B-').replace('Ri-', 'Ri-')
            
            ax.errorbar(timestamps, means, yerr=stds, 
                       color=colors.get(model_name, 'blue'), linewidth=2,
                       label=display_name, marker='s', markersize=5, capsize=4, 
                       capthick=1.5, alpha=0.8, zorder=80)
        
        ax.set_ylabel(var_label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9, framealpha=0.95, ncol=2)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', labelsize=10)
        ax.ticklabel_format(style='plain', axis='x')
        ax.grid(True, which='minor', alpha=0.15, linestyle=':')
        ax.minorticks_on()
    
    plt.tight_layout()
    
    for ext in ['png', 'pdf']:
        output_file = output_dir / f'timeseries_publication.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"✓ Saved publication time series")


def plot_physics_correlations(physics_series: List[Dict], output_dir: Path):
    """Plot physics quantity correlations."""
    
    output_dir = Path(output_dir)
    
    logger.info("\nCreating physics correlation plots...")
    
    if not physics_series:
        logger.warning("  No physics data available")
        return
    
    # Extract time series
    times = [d['timestamp'] for d in physics_series]
    times_hr = (np.array(times) - times[0]) / 3600.0
    
    # Key quantities
    quantities = ['ri_mean', 'turbulent_intensity', 'w_max', 'frac_unstable']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for ax, qty in zip(axes, quantities):
        if qty in physics_series[0]['physics']:
            values = [d['physics'][qty] for d in physics_series]
            ax.plot(times_hr, values, linewidth=2.5, marker='o', markersize=5, color='#2E86AB')
            ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
            ax.set_ylabel(qty.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_title(qty.replace('_', ' ').title(), fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'physics_evolution.png'
    plt.savefig(output_file, dpi=250, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved: {output_file.name}")


# ===================================================================
#  COMPARISON TABLES
# ===================================================================

def create_comparison_table(all_metrics: Dict, output_dir: Path):
    """Create comprehensive comparison table."""
    
    output_dir = Path(output_dir)
    
    rows = []
    for model_name, metrics in all_metrics.items():
        # Parse model type
        if model_name.startswith('Baseline-'):
            model_type = 'Baseline'
            arch = model_name.replace('Baseline-', '')
        elif model_name.startswith('Ri-'):
            model_type = 'Ri-Conditioned'
            arch = model_name.replace('Ri-', '')
        else:
            model_type = 'Unknown'
            arch = model_name
        
        row = {
            'Type': model_type,
            'Architecture': arch,
            'Visc_R2': metrics['visc_coeff']['r2_score'],
            'Visc_RMSE': metrics['visc_coeff']['rmse'],
            'Visc_MAE': metrics['visc_coeff']['mae'],
            'Diff_R2': metrics['diff_coeff']['r2_score'],
            'Diff_RMSE': metrics['diff_coeff']['rmse'],
            'Diff_MAE': metrics['diff_coeff']['mae'],
            'Rich_R2': metrics['richardson']['r2_score'],
            'Rich_RMSE': metrics['richardson']['rmse'],
            'Regime_Acc': metrics['regime']['accuracy'] if 'regime' in metrics else np.nan
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort
    df['sort_key'] = df['Type'].map({'Baseline': 0, 'Ri-Conditioned': 1})
    df = df.sort_values(['sort_key', 'Architecture']).drop('sort_key', axis=1)
    
    # Save
    csv_file = output_dir / 'comparison_table.csv'
    df.to_csv(csv_file, index=False, float_format='%.4f')
    logger.info(f"✓ Saved: {csv_file.name}")
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("COMPARISON SUMMARY")
    logger.info(f"{'='*80}\n")
    print(df.to_string(index=False))
    logger.info(f"\n{'='*80}\n")
    
    return df


def plot_improvement_heatmap(all_metrics: Dict, output_dir: Path):
    """Heatmap showing Ri-conditioned improvement over baseline."""
    
    output_dir = Path(output_dir)
    
    # Check if we have both types
    has_baseline = any(k.startswith('Baseline-') for k in all_metrics.keys())
    has_ri = any(k.startswith('Ri-') for k in all_metrics.keys())
    
    if not (has_baseline and has_ri):
        logger.info("Skipping improvement heatmap (need both baseline and Ri-conditioned)")
        return
    
    architectures = ['MLP', 'ResMLP', 'TabTransformer']
    variables = ['visc_coeff', 'diff_coeff', 'richardson']
    var_labels = ['Viscosity', 'Diffusivity', 'Richardson']
    
    improvements = np.zeros((len(architectures), len(variables)))
    
    for i, arch in enumerate(architectures):
        baseline_key = f'Baseline-{arch}'
        ri_key = f'Ri-{arch}'
        
        if baseline_key not in all_metrics or ri_key not in all_metrics:
            continue
        
        for j, var in enumerate(variables):
            baseline_r2 = all_metrics[baseline_key][var]['r2_score']
            ri_r2 = all_metrics[ri_key][var]['r2_score']
            improvements[i, j] = ri_r2 - baseline_r2
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(improvements, cmap='RdYlGn', aspect='auto', 
                   vmin=-0.1, vmax=0.1, interpolation='nearest')
    
    # Labels
    ax.set_xticks(range(len(variables)))
    ax.set_xticklabels(var_labels, fontsize=12)
    ax.set_yticks(range(len(architectures)))
    ax.set_yticklabels(architectures, fontsize=12)
    
    # Add values
    for i in range(len(architectures)):
        for j in range(len(variables)):
            val = improvements[i, j]
            color = 'white' if abs(val) > 0.05 else 'black'
            ax.text(j, i, f'{val:+.4f}', ha='center', va='center',
                   color=color, fontweight='bold', fontsize=11)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('ΔR² (Ri-Conditioned - Baseline)', 
                   fontsize=11, fontweight='bold')
    
    ax.set_title('Impact of Richardson Conditioning on Model Performance', 
                fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Variable', fontsize=12, fontweight='bold')
    ax.set_ylabel('Architecture', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = output_dir / 'ri_conditioning_improvement.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved improvement heatmap: {output_file.name}")


# ===================================================================
#  MAIN FUNCTION
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Complete Model Comparison with ALL Visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Mode
    parser.add_argument('--mode', type=str, required=True,
                       choices=['single', 'timeseries'])
    
    # Data
    parser.add_argument('--nc-file', type=Path,
                       help='NetCDF file (single mode)')
    parser.add_argument('--data-dir', type=Path,
                       help='Directory with NetCDF files (timeseries mode)')
    parser.add_argument('--output', type=Path, required=True)
    
    # BASELINE model paths
    parser.add_argument('--baseline-mlp', type=Path, default=None)
    parser.add_argument('--baseline-resmlp', type=Path, default=None)
    parser.add_argument('--baseline-tabtransformer', type=Path, default=None)
    
    # RI-CONDITIONED model paths
    parser.add_argument('--ri-mlp', type=Path, default=None)
    parser.add_argument('--ri-resmlp', type=Path, default=None)
    parser.add_argument('--ri-tabtransformer', type=Path, default=None)
    
    # Scalers and params
    parser.add_argument('--scaler-dir', type=Path, required=True)
    parser.add_argument('--time-idx', type=int, default=0)
    parser.add_argument('--k-min', type=int, default=2)
    parser.add_argument('--k-max', type=int, default=30)
    parser.add_argument('--n-workers', type=int, default=None)
    parser.add_argument('--extract-physics', action='store_true',
                       help='Extract physical quantities')
    
    args = parser.parse_args()
    
    # Build model paths
    baseline_paths = {}
    if args.baseline_mlp:
        baseline_paths['MLP'] = args.baseline_mlp
    if args.baseline_resmlp:
        baseline_paths['ResMLP'] = args.baseline_resmlp
    if args.baseline_tabtransformer:
        baseline_paths['TabTransformer'] = args.baseline_tabtransformer
    
    ri_paths = {}
    if args.ri_mlp:
        ri_paths['MLP'] = args.ri_mlp
    if args.ri_resmlp:
        ri_paths['ResMLP'] = args.ri_resmlp
    if args.ri_tabtransformer:
        ri_paths['TabTransformer'] = args.ri_tabtransformer
    
    if not baseline_paths and not ri_paths:
        parser.error("Must provide at least one model")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize engine
    logger.info(f"\n{'='*80}")
    logger.info("INITIALIZING UNIFIED INFERENCE ENGINE")
    logger.info(f"{'='*80}\n")
    
    engine = UnifiedInferenceEngine(
        baseline_paths=baseline_paths if baseline_paths else None,
        ri_paths=ri_paths if ri_paths else None,
        scaler_dir=args.scaler_dir,
        n_workers=args.n_workers
    )
    
    # ===================================================================
    #  SINGLE MODE
    # ===================================================================
    
    if args.mode == 'single':
        logger.info(f"\n{'='*80}")
        logger.info(f"SINGLE FILE MODE")
        logger.info(f"File: {args.nc_file}")
        logger.info(f"{'='*80}\n")
        
        # Inference
        predictions = engine.predict_3d_domain(
            args.nc_file, args.time_idx, args.k_min, args.k_max
        )
        
        # Truth
        truth = extract_truth_from_netcdf(
            args.nc_file, args.time_idx, args.k_min, args.k_max
        )
        
        # Metrics
        logger.info(f"\n{'='*80}")
        logger.info("CALCULATING METRICS")
        logger.info(f"{'='*80}\n")
        
        all_metrics = {}
        for name, preds in predictions.items():
            if name == 'shared':
                continue
            all_metrics[name] = calculate_3d_metrics(preds, truth)

        # ADD NON-ZERO METRICS
        logger.info(f"\n{'='*80}")
        logger.info("CALCULATING NON-ZERO METRICS")
        logger.info(f"{'='*80}\n")

        all_nonzero_metrics = {}
        for name, preds in predictions.items():
            if name == 'shared':
                continue
            all_nonzero_metrics[name] = calculate_nonzero_metrics(preds, truth)

        with open(output_dir / 'nonzero_metrics.json', 'w') as f:
            json.dump(all_nonzero_metrics, f, indent=2, cls=NumpyEncoder)
        logger.info(f"✓ Saved: nonzero_metrics.json")

        # Create non-zero comparison table
        create_nonzero_comparison_table(all_nonzero_metrics, output_dir)
        
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=2, cls=NumpyEncoder)
        logger.info(f"✓ Saved: metrics.json")
        
        # Height-wise
        heightwise = calculate_heightwise_metrics(predictions, truth)
        with open(output_dir / 'heightwise_metrics.json', 'w') as f:
            json.dump(heightwise, f, indent=2, cls=NumpyEncoder)
        logger.info(f"✓ Saved: heightwise_metrics.json")
        
        # Spatial
        spatial = diagnose_spatial_patterns(predictions, truth, output_dir)
        
        # Create comparison table
        create_comparison_table(all_metrics, output_dir)
        
        # Improvement heatmap (if both types)
        plot_improvement_heatmap(all_metrics, output_dir)
        
        # Aggregate for plotting
        agg_preds = {}
        agg_truth = {}
        for var in ['visc_coeff', 'diff_coeff', 'richardson']:
            agg_truth[var] = truth[var].flatten()
            for name in all_metrics.keys():
                if name not in agg_preds:
                    agg_preds[name] = {}
                agg_preds[name][var] = predictions[name][var].flatten()
        
        # ALL VISUALIZATIONS
        logger.info(f"\n{'='*80}")
        logger.info("CREATING ALL VISUALIZATIONS")
        logger.info(f"{'='*80}\n")
        
        plot_scatter_comparison_improved(agg_preds, agg_truth, output_dir, 'single')
        plot_distributions(agg_preds, agg_truth, output_dir, 'single')
        plot_vertical_profiles(predictions, truth, output_dir)
        plot_heightwise_metrics(heightwise, truth['heights'], output_dir)

        # ADD NON-ZERO VISUALIZATIONS
        plot_nonzero_comparison(all_nonzero_metrics, output_dir)
        plot_zero_fraction_analysis(all_nonzero_metrics, output_dir)
        
        # Physics
        if args.extract_physics:
            physics = extract_physical_quantities(args.nc_file, args.time_idx)
            with open(output_dir / 'physics.json', 'w') as f:
                json.dump(physics, f, indent=2)
            logger.info(f"✓ Saved: physics.json")
        
        logger.info(f"\n{'='*80}")
        logger.info("✅ SINGLE MODE COMPLETE")
        logger.info(f"Results: {output_dir}")
        logger.info(f"{'='*80}\n")
    
    # ===================================================================
    #  TIME SERIES MODE
    # ===================================================================
    
    elif args.mode == 'timeseries':
        logger.info(f"\n{'='*80}")
        logger.info(f"TIME SERIES MODE")
        logger.info(f"Directory: {args.data_dir}")
        logger.info(f"{'='*80}\n")
        
        files = sorted(
            args.data_dir.glob('*.nc'),
            key=lambda p: int(re.search(r'(\d+)\.nc$', p.name).group(1))
        )
        
        if not files:
            logger.error(f"No files found in {args.data_dir}")
            return
        
        logger.info(f"Found {len(files)} files\n")
        
        # Storage
        model_names = list(engine.models.keys())
        agg_preds = {name: {'visc_coeff': [], 'diff_coeff': [], 'richardson': [], 'regime': []} 
                     for name in model_names}
        agg_truth = {'visc_coeff': [], 'diff_coeff': [], 'richardson': [], 'regime': []}
        
        series_data = []
        physics_series = []
        
        last_predictions = None
        last_truth = None
        
        # Process files
        for idx, nc_file in enumerate(files):
            logger.info(f"{'='*70}")
            logger.info(f"File {idx+1}/{len(files)}: {nc_file.name}")
            logger.info(f"{'='*70}")
            
            timestamp = int(re.search(r'(\d+)\.nc$', nc_file.name).group(1))
            
            predictions = engine.predict_3d_domain(
                nc_file, args.time_idx, args.k_min, args.k_max
            )
            truth = extract_truth_from_netcdf(
                nc_file, args.time_idx, args.k_min, args.k_max
            )
            
            last_predictions = predictions
            last_truth = truth
            
            # Time series data
            timestep_series = {
                'timestamp': timestamp,
                'predictions': {},
                'truth': {}
            }
            
            for name in model_names:
                preds = predictions[name]
                timestep_series['predictions'][name] = {
                    'mean': {k: float(np.nanmean(v)) for k, v in preds.items() if isinstance(v, np.ndarray)},
                    'std': {k: float(np.nanstd(v)) for k, v in preds.items() if isinstance(v, np.ndarray)}
                }
            
            timestep_series['truth'] = {
                'mean': {k: float(np.nanmean(v)) for k, v in truth.items() if isinstance(v, np.ndarray)},
                'std': {k: float(np.nanstd(v)) for k, v in truth.items() if isinstance(v, np.ndarray)}
            }
            
            series_data.append(timestep_series)
            
            # Aggregate
            for var in ['visc_coeff', 'diff_coeff', 'richardson', 'regime']:
                agg_truth[var].append(truth[var].flatten())
                for name in model_names:
                    agg_preds[name][var].append(predictions[name][var].flatten())
            
            # Physics
            if args.extract_physics:
                physics = extract_physical_quantities(nc_file, args.time_idx)
                physics_series.append({
                    'file': nc_file.name,
                    'timestamp': timestamp,
                    'physics': physics
                })
        
        # Concatenate
        logger.info(f"\n{'='*70}")
        logger.info("Aggregating data...")
        logger.info(f"{'='*70}")
        
        for var in agg_truth.keys():
            agg_truth[var] = np.concatenate(agg_truth[var])
            for name in model_names:
                agg_preds[name][var] = np.concatenate(agg_preds[name][var])
        
        # Compute metrics
        aggregate_metrics = {}
        for name in model_names:
            temp_pred = {k: agg_preds[name][k] for k in ['visc_coeff', 'diff_coeff', 'richardson', 'regime']}
            temp_truth = {k: agg_truth[k] for k in ['visc_coeff', 'diff_coeff', 'richardson', 'regime']}
            aggregate_metrics[name] = calculate_3d_metrics(temp_pred, temp_truth)
        
        # ADD NON-ZERO METRICS FOR TIMESERIES
        logger.info(f"\n{'='*80}")
        logger.info("CALCULATING NON-ZERO METRICS (AGGREGATE)")
        logger.info(f"{'='*80}\n")

        aggregate_nonzero_metrics = {}
        for name in model_names:
            temp_pred = {k: agg_preds[name][k] for k in ['visc_coeff', 'diff_coeff', 'richardson']}
            temp_truth = {k: agg_truth[k] for k in ['visc_coeff', 'diff_coeff', 'richardson']}
            aggregate_nonzero_metrics[name] = calculate_nonzero_metrics(temp_pred, temp_truth)

        with open(output_dir / 'nonzero_metrics_aggregate.json', 'w') as f:
            json.dump(aggregate_nonzero_metrics, f, indent=2, cls=NumpyEncoder)
        logger.info(f"✓ Saved: nonzero_metrics_aggregate.json")

        create_nonzero_comparison_table(aggregate_nonzero_metrics, output_dir)

        with open(output_dir / 'metrics_aggregate.json', 'w') as f:
            json.dump(aggregate_metrics, f, indent=2, cls=NumpyEncoder)
        logger.info(f"✓ Saved: metrics_aggregate.json")
        
        # Print summary
        logger.info("\nAGGREGATE RESULTS:")
        for name, metrics in aggregate_metrics.items():
            logger.info(f"\n  {name}:")
            logger.info(f"    Visc R²: {metrics['visc_coeff']['r2_score']:.4f}")
            logger.info(f"    Diff R²: {metrics['diff_coeff']['r2_score']:.4f}")
            logger.info(f"    Rich R²: {metrics['richardson']['r2_score']:.4f}")
            if 'regime' in metrics:
                logger.info(f"    Regime Acc:  {metrics['regime']['accuracy']:.4f}")
        
        # Create tables
        create_comparison_table(aggregate_metrics, output_dir)
        plot_improvement_heatmap(aggregate_metrics, output_dir)
        
        # ALL VISUALIZATIONS
        logger.info(f"\n{'='*70}")
        logger.info("CREATING ALL VISUALIZATIONS")
        logger.info(f"{'='*70}\n")
        
        plot_time_series(series_data, output_dir)
        
        if last_predictions and last_truth:
            heightwise = calculate_heightwise_metrics(last_predictions, last_truth)
            plot_vertical_profiles(last_predictions, last_truth, output_dir)
            plot_heightwise_metrics(heightwise, last_truth['heights'], output_dir)
            diagnose_spatial_patterns(last_predictions, last_truth, output_dir)
            
        plot_nonzero_comparison(aggregate_nonzero_metrics, output_dir)
        plot_zero_fraction_analysis(aggregate_nonzero_metrics, output_dir)

        plot_scatter_comparison_improved(agg_preds, agg_truth, output_dir, 
                                        f'timeseries_{len(files)}files')
        plot_distributions(agg_preds, agg_truth, output_dir, 
                          f'timeseries_{len(files)}files')
        
        if args.extract_physics:
            with open(output_dir / 'physics_timeseries.json', 'w') as f:
                json.dump(physics_series, f, indent=2, cls=NumpyEncoder)
            plot_physics_correlations(physics_series, output_dir)
            logger.info(f"✓ Saved: physics_timeseries.json")
        
        logger.info(f"\n{'='*80}")
        logger.info("✅ TIME SERIES ANALYSIS COMPLETE")
        logger.info(f"Results: {output_dir}")
        logger.info(f"Files: {len(files)}")
        logger.info(f"Total points: {len(agg_truth['visc_coeff']):,}")
        logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    main()
