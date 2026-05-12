#!/usr/bin/env python3
"""
Comprehensive ML-SGS Analysis Suite with Full Diagnostics
==========================================================

Generates complete analysis including:
- Spatial diagnostics (autocorrelation, power spectra, mean fields)
- Temporal diagnostics (metrics time series, prediction evolution)
- Vertical profiles (full k_min to k_max domain)
- 2D slice visualizations (selected k-levels only)
- Scatter density plots (aggregate, full domain)
- Non-zero evaluation (full domain)
- Positivity constraint analysis (full domain)

DOMAIN USAGE:
- --k-min and --k-max: Define full vertical domain for ALL analysis
- --k-levels: Only for selecting which 2D slices to visualize

TIMESTAMPS:
- Extracted as raw simulation time (seconds) from filenames
- Example: RCE_diagnostic_3d_48600.nc → 48600 seconds
- X-axis shows actual simulation time in seconds

Usage:
    # Single file
    python comprehensive_analysis_suite.py \
        --mode single \
        --nc-file data.nc \
        --baseline-mlp mlp.pth --ri-mlp ri_mlp.pth \
        --scaler-dir scalers/ --output results/ \
        --k-min 0 --k-max 219 --k-levels 10 50 100 150 200
    
    # Time series
    python comprehensive_analysis_suite.py \
        --mode timeseries \
        --data-dir data_files/ \
        --baseline-mlp mlp.pth --ri-mlp ri_mlp.pth \
        --scaler-dir scalers/ --output results/ \
        --k-min 0 --k-max 219 --k-levels 10 50 100 150 200
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import logging
from typing import Dict, Optional, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import re
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib.colors import LogNorm
from scipy import signal
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Import model architectures
from train_new_coeff import UnifiedSGSCoefficientNetwork
from train_resmlp import ResMLPNetwork
from train_tab_transformer import TabTransformerNetwork
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
    NumpyEncoder
)
from run_models_comparison_with_ri_v2 import UnifiedInferenceEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="paper")


# ===================================================================
#  COLOR AND STYLE MANAGEMENT
# ===================================================================

def get_model_color_and_style(model_name: str, all_model_names: List[str]) -> Tuple[str, str]:
    """
    Get unique color and line style for each model.
    
    Returns
    -------
    color : str
        Hex color code
    style : str
        Line style ('-', '--', ':', '-.')
    """
    # Define distinct color palette (colorblind-friendly)
    colors = [
        '#E63946',  # Red
        '#2E86AB',  # Blue
        '#06A77D',  # Green
        '#F77F00',  # Orange
        '#8338EC',  # Purple
        '#A4036F',  # Magenta
        '#118AB2',  # Cyan
        '#FFB703',  # Yellow
    ]
    
    # Line styles
    styles = ['-', '--', '-.', ':']
    
    # Find index of this model
    try:
        idx = all_model_names.index(model_name)
    except ValueError:
        idx = 0
    
    # Assign color (cycle through if more than 8 models)
    color = colors[idx % len(colors)]
    
    # Assign line style (cycle through)
    style = styles[idx % len(styles)]
    
    return color, style


# ===================================================================
#  TIMESTAMP EXTRACTION
# ===================================================================

def extract_timestamp_from_filename(filename: Path) -> Optional[float]:
    """
    Extract simulation time (seconds) from NetCDF filename.
    
    Examples:
    - RCE_diagnostic_3d_48600.nc → 48600.0
    - arm_data_21600.nc → 21600.0
    - output_003600.nc → 3600.0
    
    Returns None if no number found.
    """
    name = filename.stem
    
    # Try to extract any sequence of digits before .nc
    match = re.search(r'(\d+)$', name)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    return None


def get_timestamps_from_files(nc_files: List[Path]) -> Tuple[List[float], bool]:
    """
    Extract simulation times (seconds) from list of NetCDF files.
    
    Returns
    -------
    timestamps : List[float]
        List of simulation times in seconds
    success : bool
        True if timestamps successfully extracted, False if using fallback indices
    """
    timestamps = []
    for f in nc_files:
        ts = extract_timestamp_from_filename(f)
        if ts is not None:
            timestamps.append(ts)
        else:
            timestamps.append(None)
    
    # Check if we got valid timestamps
    if all(t is not None for t in timestamps):
        logger.info(f"✓ Successfully extracted timestamps from filenames")
        logger.info(f"  Time range: {min(timestamps):.0f} to {max(timestamps):.0f} seconds")
        return timestamps, True
    else:
        logger.warning("⚠️  Could not extract timestamps from all files")
        logger.warning("   Using sequential indices instead")
        # Fallback: use indices
        timestamps = [float(i) for i in range(len(nc_files))]
        return timestamps, False


# ===================================================================
#  CONSTRAINT APPLICATION (FIXED)
# ===================================================================

def apply_physical_constraints(predictions: Dict, verbose: bool = False) -> Dict:
    """
    Apply physical constraints IN PHYSICAL SPACE (after inverse transform).
    
    Constraints:
    - visc_coeff >= 0
    - diff_coeff >= 0
    - richardson: no constraint
    """
    constrained = {}
    
    for key, values in predictions.items():
        if key in ['visc_coeff', 'diff_coeff']:
            constrained[key] = np.maximum(values, 0.0)
            
            if verbose:
                n_negative = np.sum(values < 0)
                if n_negative > 0:
                    pct = 100 * n_negative / values.size
                    logger.info(f"    {key}: {n_negative:,} ({pct:.2f}%) clipped to 0")
        else:
            constrained[key] = values.copy()
    
    return constrained


# ===================================================================
#  SPATIAL DIAGNOSTICS
# ===================================================================

def compute_spatial_autocorrelation(field: np.ndarray, max_lag: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D spatial autocorrelation for a 3D field (x, y, z).
    Average over z-levels.
    """
    nx, ny, nz = field.shape
    autocorr_x = []
    autocorr_y = []
    
    for k in range(nz):
        slice_2d = field[:, :, k]
        
        # Remove mean
        slice_centered = slice_2d - np.nanmean(slice_2d)
        
        # Compute autocorrelation along x
        corr_x = []
        for lag in range(max_lag):
            if lag == 0:
                corr_x.append(1.0)
            else:
                shifted = np.roll(slice_centered, lag, axis=0)
                valid = ~(np.isnan(slice_centered) | np.isnan(shifted))
                if np.sum(valid) > 0:
                    corr = np.corrcoef(slice_centered[valid], shifted[valid])[0, 1]
                    corr_x.append(corr if not np.isnan(corr) else 0.0)
                else:
                    corr_x.append(0.0)
        
        # Compute autocorrelation along y
        corr_y = []
        for lag in range(max_lag):
            if lag == 0:
                corr_y.append(1.0)
            else:
                shifted = np.roll(slice_centered, lag, axis=1)
                valid = ~(np.isnan(slice_centered) | np.isnan(shifted))
                if np.sum(valid) > 0:
                    corr = np.corrcoef(slice_centered[valid], shifted[valid])[0, 1]
                    corr_y.append(corr if not np.isnan(corr) else 0.0)
                else:
                    corr_y.append(0.0)
        
        autocorr_x.append(corr_x)
        autocorr_y.append(corr_y)
    
    # Average over z-levels
    autocorr_x = np.nanmean(autocorr_x, axis=0)
    autocorr_y = np.nanmean(autocorr_y, axis=0)
    
    return autocorr_x, autocorr_y


def compute_power_spectrum_2d(field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D power spectrum averaged over z-levels.
    """
    nx, ny, nz = field.shape
    spectra = []
    
    for k in range(nz):
        slice_2d = field[:, :, k]
        
        # Remove NaNs
        if np.any(np.isnan(slice_2d)):
            slice_2d = np.nan_to_num(slice_2d, nan=0.0)
        
        # 2D FFT
        fft_2d = np.fft.fft2(slice_2d)
        power = np.abs(fft_2d)**2
        
        # Radial average
        kx = np.fft.fftfreq(nx)
        ky = np.fft.fftfreq(ny)
        kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
        k_radial = np.sqrt(kx_grid**2 + ky_grid**2)
        
        # Bin by radial wavenumber
        k_bins = np.linspace(0, 0.5, 50)
        power_binned = []
        k_centers = []
        
        for i in range(len(k_bins) - 1):
            mask = (k_radial >= k_bins[i]) & (k_radial < k_bins[i+1])
            if np.sum(mask) > 0:
                power_binned.append(np.mean(power[mask]))
                k_centers.append((k_bins[i] + k_bins[i+1]) / 2)
        
        spectra.append(power_binned)
    
    # Average over z
    spectrum = np.nanmean(spectra, axis=0)
    wavenumbers = np.array(k_centers)
    
    return wavenumbers, spectrum


def compute_mean_field_structure(field: np.ndarray) -> Dict:
    """
    Compute mean field statistics.
    """
    return {
        'horizontal_mean': np.nanmean(field, axis=(0, 1)),  # Mean over x,y for each z
        'vertical_mean': np.nanmean(field, axis=2),         # Mean over z for each x,y
        'global_mean': np.nanmean(field),
        'global_std': np.nanstd(field),
        'vertical_variance': np.nanvar(field, axis=2)
    }


def diagnose_spatial_patterns(predictions: Dict, truth: Dict, output_dir: Path, 
                               model_name: str, time_idx: Optional[int] = None) -> Dict:
    """
    Comprehensive spatial diagnostics on FULL DOMAIN (k_min to k_max).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"SPATIAL DIAGNOSTICS: {model_name}")
    if time_idx is not None:
        logger.info(f"Time index: {time_idx}")
    logger.info(f"Domain: Full vertical extent (all z-levels)")
    logger.info(f"{'='*60}\n")
    
    results = {}
    
    for var in ['visc_coeff', 'diff_coeff']:
        if var not in predictions or var not in truth:
            continue
        
        logger.info(f"Processing {var}...")
        
        pred = predictions[var]
        true = truth[var]
        
        logger.info(f"  Field shape: {pred.shape}")
        
        # Autocorrelation
        logger.info("  Computing autocorrelation...")
        pred_autocorr_x, pred_autocorr_y = compute_spatial_autocorrelation(pred)
        true_autocorr_x, true_autocorr_y = compute_spatial_autocorrelation(true)
        
        # Power spectrum
        logger.info("  Computing power spectrum...")
        pred_k, pred_spectrum = compute_power_spectrum_2d(pred)
        true_k, true_spectrum = compute_power_spectrum_2d(true)
        
        # Mean field structure
        logger.info("  Computing mean field structure...")
        pred_mean = compute_mean_field_structure(pred)
        true_mean = compute_mean_field_structure(true)
        
        results[var] = {
            'autocorrelation': {
                'pred_x': pred_autocorr_x.tolist(),
                'pred_y': pred_autocorr_y.tolist(),
                'true_x': true_autocorr_x.tolist(),
                'true_y': true_autocorr_y.tolist()
            },
            'power_spectrum': {
                'wavenumbers': pred_k.tolist(),
                'pred_power': pred_spectrum.tolist(),
                'true_power': true_spectrum.tolist()
            },
            'mean_field': {
                'prediction': {
                    'horizontal_mean': pred_mean['horizontal_mean'].tolist(),
                    'global_mean': float(pred_mean['global_mean']),
                    'global_std': float(pred_mean['global_std'])
                },
                'truth': {
                    'horizontal_mean': true_mean['horizontal_mean'].tolist(),
                    'global_mean': float(true_mean['global_mean']),
                    'global_std': float(true_mean['global_std'])
                }
            }
        }
    
    # Save JSON
    suffix = f'_t{time_idx}' if time_idx is not None else ''
    json_file = output_dir / f'spatial_diagnostics_{model_name}{suffix}.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"✓ Saved: {json_file.name}\n")
    
    return results


def plot_spatial_diagnostics(all_spatial_results: Dict, output_dir: Path):
    """
    Visualize spatial diagnostics for all models.
    """
    output_dir = Path(output_dir)
    
    logger.info("\nCreating spatial diagnostic plots...")
    
    model_names = list(all_spatial_results.keys())
    
    for var in ['visc_coeff', 'diff_coeff']:
        # Check if all models have this variable
        if not all(var in all_spatial_results[m] for m in model_names):
            continue
        
        var_label = var.replace('_coeff', '').upper()
        
        # 1. Autocorrelation plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # X-direction
        ax = axes[0]
        for model_name in model_names:
            data = all_spatial_results[model_name][var]['autocorrelation']
            lags = np.arange(len(data['pred_x']))
            
            color, style = get_model_color_and_style(model_name, model_names)
            
            ax.plot(lags, data['pred_x'], style, color=color, alpha=0.8, 
                   label=model_name, linewidth=2.5)
        
        # Truth
        truth_data = all_spatial_results[model_names[0]][var]['autocorrelation']
        ax.plot(lags, truth_data['true_x'], 'k-', linewidth=3.5, alpha=0.9, label='MONC Truth')
        
        ax.set_xlabel('Lag (grid points)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Autocorrelation', fontsize=11, fontweight='bold')
        ax.set_title('X-Direction', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.2, 1.05])
        
        # Y-direction
        ax = axes[1]
        for model_name in model_names:
            data = all_spatial_results[model_name][var]['autocorrelation']
            lags = np.arange(len(data['pred_y']))
            
            color, style = get_model_color_and_style(model_name, model_names)
            
            ax.plot(lags, data['pred_y'], style, color=color, alpha=0.8, 
                   label=model_name, linewidth=2.5)
        
        ax.plot(lags, truth_data['true_y'], 'k-', linewidth=3.5, alpha=0.9, label='MONC Truth')
        
        ax.set_xlabel('Lag (grid points)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Autocorrelation', fontsize=11, fontweight='bold')
        ax.set_title('Y-Direction', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.2, 1.05])
        
        fig.suptitle(f'{var_label} - Spatial Autocorrelation (Full Domain)', 
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        output_file = output_dir / f'spatial_autocorrelation_{var}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"  ✓ Saved: {output_file.name}")
        
        # 2. Power spectrum plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model_name in model_names:
            data = all_spatial_results[model_name][var]['power_spectrum']
            k = np.array(data['wavenumbers'])
            power = np.array(data['pred_power'])
            
            color, style = get_model_color_and_style(model_name, model_names)
            
            ax.loglog(k, power, style, color=color, alpha=0.8, 
                     label=model_name, linewidth=2.5)
        
        # Truth
        truth_data = all_spatial_results[model_names[0]][var]['power_spectrum']
        k_true = np.array(truth_data['wavenumbers'])
        power_true = np.array(truth_data['true_power'])
        ax.loglog(k_true, power_true, 'k-', linewidth=3.5, alpha=0.9, label='MONC Truth')
        
        # Reference line (k^-5/3 for turbulence)
        k_ref = k_true[k_true > 0]
        if len(k_ref) > 0:
            power_ref = k_ref**(-5/3) * power_true[0] / k_ref[0]**(-5/3)
            ax.loglog(k_ref, power_ref, 'k:', linewidth=2, alpha=0.4, label='k⁻⁵ᐟ³')
        
        ax.set_xlabel('Wavenumber k', fontsize=11, fontweight='bold')
        ax.set_ylabel('Power Spectral Density', fontsize=11, fontweight='bold')
        ax.set_title(f'{var_label} - 2D Power Spectrum (Full Domain)', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        output_file = output_dir / f'power_spectrum_{var}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"  ✓ Saved: {output_file.name}")


# ===================================================================
#  VERTICAL PROFILES
# ===================================================================

def compute_vertical_profiles(predictions: Dict, truth: Dict, model_name: str) -> Dict:
    """
    Compute vertical profiles (horizontally averaged) for FULL DOMAIN.
    """
    profiles = {}
    
    for var in ['visc_coeff', 'diff_coeff']:
        if var not in predictions or var not in truth:
            continue
        
        pred = predictions[var]
        true = truth[var]
        
        # Average over x and y
        pred_profile = np.nanmean(pred, axis=(0, 1))
        true_profile = np.nanmean(true, axis=(0, 1))
        
        # Compute error
        error = pred_profile - true_profile
        rel_error = error / (np.abs(true_profile) + 1e-10)
        
        profiles[var] = {
            'prediction': pred_profile.tolist(),
            'truth': true_profile.tolist(),
            'error': error.tolist(),
            'relative_error': rel_error.tolist()
        }
    
    return profiles


def plot_vertical_profiles(all_profiles: Dict, output_dir: Path, domain_info: Dict = None):
    """
    Plot vertical profiles for all models.
    """
    output_dir = Path(output_dir)
    
    logger.info("\nCreating vertical profile plots...")
    
    model_names = list(all_profiles.keys())
    
    for var in ['visc_coeff', 'diff_coeff']:
        if not all(var in all_profiles[m] for m in model_names):
            continue
        
        var_label = var.replace('_coeff', '').title()
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        
        # Get vertical levels
        n_levels = len(all_profiles[model_names[0]][var]['truth'])
        z_levels = np.arange(n_levels)
        
        # Add k_min offset if provided
        if domain_info and 'k_min' in domain_info:
            z_levels = z_levels + domain_info['k_min']
        
        # 1. Absolute values
        ax = axes[0]
        for model_name in model_names:
            data = all_profiles[model_name][var]
            pred = np.array(data['prediction'])
            
            color, style = get_model_color_and_style(model_name, model_names)
            
            ax.plot(pred, z_levels, style, color=color, alpha=0.8, 
                   label=model_name, linewidth=2.5)
        
        # Truth
        truth = np.array(all_profiles[model_names[0]][var]['truth'])
        ax.plot(truth, z_levels, 'k-', linewidth=3.5, alpha=0.9, label='MONC Truth')
        
        ax.set_xlabel(f'{var_label}', fontsize=11, fontweight='bold')
        ax.set_ylabel('Vertical Level (k)', fontsize=11, fontweight='bold')
        ax.set_title('Vertical Profiles', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # 2. Absolute error
        ax = axes[1]
        for model_name in model_names:
            data = all_profiles[model_name][var]
            error = np.array(data['error'])
            
            color, style = get_model_color_and_style(model_name, model_names)
            
            ax.plot(error, z_levels, style, color=color, alpha=0.8, 
                   label=model_name, linewidth=2.5)
        
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Error (Pred - Truth)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Vertical Level (k)', fontsize=11, fontweight='bold')
        ax.set_title('Vertical Error Profile', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # 3. Relative error (%)
        ax = axes[2]
        for model_name in model_names:
            data = all_profiles[model_name][var]
            rel_error = np.array(data['relative_error']) * 100
            
            color, style = get_model_color_and_style(model_name, model_names)
            
            ax.plot(rel_error, z_levels, style, color=color, alpha=0.8, 
                   label=model_name, linewidth=2.5)
        
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Relative Error (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Vertical Level (k)', fontsize=11, fontweight='bold')
        ax.set_title('Relative Error Profile', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Add domain info to title
        title = f'{var_label} - Vertical Profiles'
        if domain_info:
            title += f' (k={domain_info.get("k_min", 0)} to {domain_info.get("k_max", n_levels-1)})'
        fig.suptitle(title, fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        output_file = output_dir / f'vertical_profiles_{var}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"  ✓ Saved: {output_file.name}")


# ===================================================================
#  TEMPORAL DIAGNOSTICS (TIME SERIES)
# ===================================================================

def compute_metrics_for_timestep(predictions: Dict, truth: Dict) -> Dict:
    """
    Compute metrics for a single time step on FULL DOMAIN.
    """
    metrics = {}
    
    for var in ['visc_coeff', 'diff_coeff']:
        if var not in predictions or var not in truth:
            continue
        
        pred_flat = predictions[var].flatten()
        true_flat = truth[var].flatten()
        
        valid = ~(np.isnan(pred_flat) | np.isnan(true_flat) | 
                 np.isinf(pred_flat) | np.isinf(true_flat))
        
        pred_valid = pred_flat[valid]
        true_valid = true_flat[valid]
        
        if len(pred_valid) > 0:
            r2 = r2_score(true_valid, pred_valid)
            rmse = np.sqrt(mean_squared_error(true_valid, pred_valid))
            mae = mean_absolute_error(true_valid, pred_valid)
            bias = np.mean(pred_valid - true_valid)
            
            # Variance ratio
            var_pred = np.var(pred_valid)
            var_true = np.var(true_valid)
            var_ratio = var_pred / var_true if var_true > 0 else np.nan
            
            metrics[var] = {
                'r2': float(r2),
                'rmse': float(rmse),
                'mae': float(mae),
                'bias': float(bias),
                'variance_ratio': float(var_ratio),
                'mean_pred': float(np.mean(pred_valid)),
                'mean_truth': float(np.mean(true_valid)),
                'std_pred': float(np.std(pred_valid)),
                'std_truth': float(np.std(true_valid))
            }
        else:
            metrics[var] = None
    
    return metrics


def plot_metrics_timeseries(metrics_timeseries: Dict, timestamps: List[float], 
                            output_dir: Path, use_real_time: bool = True):
    """
    Plot time evolution of metrics with simulation times (seconds).
    """
    output_dir = Path(output_dir)
    
    logger.info("\nCreating metrics time series plots...")
    
    model_names = list(metrics_timeseries.keys())
    n_timesteps = len(metrics_timeseries[model_names[0]])
    
    for var in ['visc_coeff', 'diff_coeff']:
        var_label = var.replace('_coeff', '').upper()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Use simulation times directly (seconds)
        x_data = timestamps[:n_timesteps]
        
        # 1. R²
        ax = axes[0, 0]
        for model_name in model_names:
            r2_values = []
            x_valid = []
            for t in range(n_timesteps):
                if metrics_timeseries[model_name][t][var] is not None:
                    r2_values.append(metrics_timeseries[model_name][t][var]['r2'])
                    x_valid.append(x_data[t])
            
            color, style = get_model_color_and_style(model_name, model_names)
            
            ax.plot(x_valid, r2_values, style, color=color, alpha=0.8, 
                   label=model_name, linewidth=2.5, marker='o', markersize=5)
        
        ax.set_xlabel('Simulation Time (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('R²', fontsize=11, fontweight='bold')
        ax.set_title('R² Evolution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # 2. RMSE
        ax = axes[0, 1]
        for model_name in model_names:
            rmse_values = []
            x_valid = []
            for t in range(n_timesteps):
                if metrics_timeseries[model_name][t][var] is not None:
                    rmse_values.append(metrics_timeseries[model_name][t][var]['rmse'])
                    x_valid.append(x_data[t])
            
            color, style = get_model_color_and_style(model_name, model_names)
            
            ax.plot(x_valid, rmse_values, style, color=color, alpha=0.8, 
                   label=model_name, linewidth=2.5, marker='o', markersize=5)
        
        ax.set_xlabel('Simulation Time (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('RMSE', fontsize=11, fontweight='bold')
        ax.set_title('RMSE Evolution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # 3. Bias
        ax = axes[1, 0]
        for model_name in model_names:
            bias_values = []
            x_valid = []
            for t in range(n_timesteps):
                if metrics_timeseries[model_name][t][var] is not None:
                    bias_values.append(metrics_timeseries[model_name][t][var]['bias'])
                    x_valid.append(x_data[t])
            
            color, style = get_model_color_and_style(model_name, model_names)
            
            ax.plot(x_valid, bias_values, style, color=color, alpha=0.8, 
                   label=model_name, linewidth=2.5, marker='o', markersize=5)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Simulation Time (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Bias', fontsize=11, fontweight='bold')
        ax.set_title('Bias Evolution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # 4. Variance ratio
        ax = axes[1, 1]
        for model_name in model_names:
            var_ratio_values = []
            x_valid = []
            for t in range(n_timesteps):
                if metrics_timeseries[model_name][t][var] is not None:
                    var_ratio_values.append(metrics_timeseries[model_name][t][var]['variance_ratio'])
                    x_valid.append(x_data[t])
            
            color, style = get_model_color_and_style(model_name, model_names)
            
            ax.plot(x_valid, var_ratio_values, style, color=color, alpha=0.8, 
                   label=model_name, linewidth=2.5, marker='o', markersize=5)
        
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Simulation Time (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Variance Ratio', fontsize=11, fontweight='bold')
        ax.set_title('Variance Ratio Evolution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'{var_label} - Metrics Time Series (Full Domain)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_file = output_dir / f'metrics_timeseries_{var}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"  ✓ Saved: {output_file.name}")

def plot_prediction_timeseries(prediction_timeseries: Dict, truth_timeseries: Dict, 
                               timestamps: List[float], output_dir: Path, 
                               use_real_time: bool = True):
    """
    Plot time evolution of DOMAIN-AVERAGED predictions.
    
    Shows how the spatial mean of each coefficient evolves over time,
    comparing model predictions to MONC truth.
    """
    output_dir = Path(output_dir)
    
    logger.info("\nCreating prediction time series plots (domain-averaged)...")
    
    model_names = list(prediction_timeseries.keys())
    n_timesteps = len(truth_timeseries['visc_coeff'])
    
    # Use simulation times directly
    x_data = timestamps[:n_timesteps]
    
    for var in ['visc_coeff', 'diff_coeff']:
        var_label = var.replace('_coeff', '').title()
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Panel 1: Absolute values
        ax = axes[0]
        
        # Calculate domain-averaged truth time series
        truth_mean_ts = [np.nanmean(truth_timeseries[var][t]) for t in range(n_timesteps)]
        
        # Plot truth with thick black line
        ax.plot(x_data, truth_mean_ts, 'k-', linewidth=3.5, alpha=0.9, 
               label='MONC Truth', marker='s', markersize=7, markevery=max(1, n_timesteps//10))
        
        # Plot each model's domain-averaged predictions
        for model_name in model_names:
            pred_mean_ts = [np.nanmean(prediction_timeseries[model_name][t][var]) 
                           for t in range(n_timesteps)]
            
            color, style = get_model_color_and_style(model_name, model_names)
            
            ax.plot(x_data, pred_mean_ts, style, color=color, alpha=0.8, 
                   label=model_name, linewidth=2.5, marker='o', markersize=5,
                   markevery=max(1, n_timesteps//10))
        
        ax.set_xlabel('Simulation Time (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Domain-Averaged {var_label}', fontsize=11, fontweight='bold')
        ax.set_title('Domain-Averaged Values Over Time', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Errors (prediction - truth)
        ax = axes[1]
        
        for model_name in model_names:
            pred_mean_ts = np.array([np.nanmean(prediction_timeseries[model_name][t][var]) 
                                    for t in range(n_timesteps)])
            truth_mean_ts_array = np.array(truth_mean_ts)
            
            error_ts = pred_mean_ts - truth_mean_ts_array
            
            color, style = get_model_color_and_style(model_name, model_names)
            
            ax.plot(x_data, error_ts, style, color=color, alpha=0.8, 
                   label=model_name, linewidth=2.5, marker='o', markersize=5,
                   markevery=max(1, n_timesteps//10))
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.set_xlabel('Simulation Time (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Error (Pred - Truth)', fontsize=11, fontweight='bold')
        ax.set_title('Domain-Averaged Error Over Time', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'{var_label} - Domain-Averaged Temporal Evolution (Full Domain)', 
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        output_file = output_dir / f'prediction_timeseries_{var}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"  ✓ Saved: {output_file.name}")


def plot_prediction_timeseries_old(prediction_timeseries: Dict, truth_timeseries: Dict, 
                               timestamps: List[float], output_dir: Path, 
                               sample_points: int = 5, use_real_time: bool = True):
    """
    Plot time evolution of predictions at selected spatial points.
    """
    output_dir = Path(output_dir)
    
    logger.info("\nCreating prediction time series plots...")
    
    model_names = list(prediction_timeseries.keys())
    n_timesteps = len(truth_timeseries['visc_coeff'])
    
    for var in ['visc_coeff', 'diff_coeff']:
        var_label = var.replace('_coeff', '').title()
        
        # Select random spatial points
        first_timestep_shape = truth_timeseries[var][0].shape
        np.random.seed(42)
        sample_indices = [
            (np.random.randint(first_timestep_shape[0]),
             np.random.randint(first_timestep_shape[1]),
             np.random.randint(first_timestep_shape[2]))
            for _ in range(sample_points)
        ]
        
        fig, axes = plt.subplots(sample_points, 1, figsize=(14, 3*sample_points))
        if sample_points == 1:
            axes = [axes]
        
        # Use simulation times directly
        x_data = timestamps[:n_timesteps]
        
        for idx, (i, j, k) in enumerate(sample_indices):
            ax = axes[idx]
            
            # Extract time series for this point
            truth_ts = [truth_timeseries[var][t][i, j, k] for t in range(n_timesteps)]
            
            # Plot truth with distinct thick black line
            ax.plot(x_data, truth_ts, 'k-', linewidth=3, alpha=0.9, 
                   label='MONC Truth', marker='s', markersize=7, markevery=max(1, n_timesteps//10))
            
            for model_name in model_names:
                pred_ts = [prediction_timeseries[model_name][t][var][i, j, k] 
                          for t in range(n_timesteps)]
                
                color, style = get_model_color_and_style(model_name, model_names)
                
                ax.plot(x_data, pred_ts, style, color=color, alpha=0.8, 
                       label=model_name, linewidth=2.5, marker='o', markersize=5,
                       markevery=max(1, n_timesteps//10))
            
            ax.set_xlabel('Simulation Time (s)', fontsize=10, fontweight='bold')
            ax.set_ylabel(f'{var_label}', fontsize=10, fontweight='bold')
            ax.set_title(f'Point ({i}, {j}, {k})', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='best', framealpha=0.9, ncol=2)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'{var_label} - Temporal Evolution at Sample Points (Full Domain)', 
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        output_file = output_dir / f'prediction_timeseries_{var}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"  ✓ Saved: {output_file.name}")


# ===================================================================
#  2D SLICE VISUALIZATIONS
# ===================================================================

def plot_2d_slices(predictions: Dict, truth: Dict, output_dir: Path, 
                   model_name: str, k_levels: List[int] = [10, 50, 100, 150, 200]):
    """
    Plot 2D slices at SELECTED k-levels for visualization only.
    
    This is for visual inspection - all metrics use the full domain.
    
    Parameters
    ----------
    k_levels : List[int]
        Specific vertical levels to visualize. These are absolute indices.
    """
    output_dir = Path(output_dir)
    slice_dir = output_dir / '2d_slices' / model_name
    slice_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n  Creating 2D slice visualizations at k-levels: {k_levels}")
    logger.info(f"  (Note: Visualization only - metrics use full domain)")
    
    for var in ['visc_coeff', 'diff_coeff']:
        if var not in predictions or var not in truth:
            continue
        
        var_label = var.replace('_coeff', '').title()
        pred = predictions[var]
        true = truth[var]
        
        for k in k_levels:
            if k >= pred.shape[2]:
                logger.warning(f"  k={k} exceeds domain (max={pred.shape[2]-1}), skipping")
                continue
            
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            
            # Common colorbar limits
            vmin = min(np.nanmin(pred[:, :, k]), np.nanmin(true[:, :, k]))
            vmax = max(np.nanmax(pred[:, :, k]), np.nanmax(true[:, :, k]))
            
            # Truth
            ax = axes[0]
            im = ax.imshow(true[:, :, k].T, origin='lower', cmap='viridis', 
                          vmin=vmin, vmax=vmax, aspect='auto')
            ax.set_title('MONC Truth', fontsize=11, fontweight='bold')
            ax.set_xlabel('X', fontsize=10)
            ax.set_ylabel('Y', fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Prediction
            ax = axes[1]
            im = ax.imshow(pred[:, :, k].T, origin='lower', cmap='viridis', 
                          vmin=vmin, vmax=vmax, aspect='auto')
            ax.set_title('Prediction', fontsize=11, fontweight='bold')
            ax.set_xlabel('X', fontsize=10)
            ax.set_ylabel('Y', fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Error
            ax = axes[2]
            error = pred[:, :, k] - true[:, :, k]
            error_max = max(abs(np.nanmin(error)), abs(np.nanmax(error)))
            im = ax.imshow(error.T, origin='lower', cmap='RdBu_r', 
                          vmin=-error_max, vmax=error_max, aspect='auto')
            ax.set_title('Error (Pred - Truth)', fontsize=11, fontweight='bold')
            ax.set_xlabel('X', fontsize=10)
            ax.set_ylabel('Y', fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            fig.suptitle(f'{model_name} - {var_label} at k={k}', 
                        fontsize=13, fontweight='bold')
            plt.tight_layout()
            
            output_file = slice_dir / f'{var}_k{k:03d}.png'
            plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
    
    logger.info(f"  ✓ Saved 2D slices to: {slice_dir}")


# ===================================================================
#  SCATTER DENSITY PLOTS (AGGREGATE)
# ===================================================================

def plot_aggregate_scatter_density(predictions: Dict, truth: Dict, output_dir: Path,
                                   model_name: str, max_points: int = 2_000_000):
    """
    Create scatter density plots from FULL DOMAIN aggregated data.
    """
    output_dir = Path(output_dir)
    
    logger.info(f"\n  Creating aggregate scatter plots for {model_name}...")
    
    for var in ['visc_coeff', 'diff_coeff']:
        if var not in predictions or var not in truth:
            continue
        
        var_label = var.replace('_coeff', '').title()
        
        # Get flattened data (handle both 3D and already-flattened)
        if predictions[var].ndim > 1:
            pred_flat = predictions[var].flatten()
            true_flat = truth[var].flatten()
        else:
            pred_flat = predictions[var]
            true_flat = truth[var]
        
        # Remove invalid
        valid = ~(np.isnan(pred_flat) | np.isnan(true_flat) | 
                 np.isinf(pred_flat) | np.isinf(true_flat))
        pred_valid = pred_flat[valid]
        true_valid = true_flat[valid]
        
        # Downsample if needed
        n_total = len(pred_valid)
        if n_total > max_points:
            logger.info(f"    {var}: Downsampling {n_total:,} → {max_points:,} points")
            indices = np.random.choice(n_total, max_points, replace=False)
            pred_valid = pred_valid[indices]
            true_valid = true_valid[indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(9, 9))
        
        data_min = np.percentile(true_valid, 1)
        data_max = np.percentile(true_valid, 99)
        
        hexbin = ax.hexbin(true_valid, pred_valid, gridsize=80, cmap='Blues',
                         norm=LogNorm(vmin=1, vmax=max(10, len(pred_valid)/50)),
                         mincnt=1, alpha=0.9, extent=(data_min, data_max, data_min, data_max))
        
        cbar = plt.colorbar(hexbin, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Count (log)', fontsize=11)
        
        ax.plot([data_min, data_max], [data_min, data_max], 
               'r--', linewidth=2.5, alpha=0.8, label='1:1 Line', zorder=10)
        
        # Metrics
        r2 = r2_score(true_valid, pred_valid)
        rmse = np.sqrt(mean_squared_error(true_valid, pred_valid))
        mae = mean_absolute_error(true_valid, pred_valid)
        bias = np.mean(pred_valid - true_valid)
        var_ratio = np.var(pred_valid) / np.var(true_valid)
        
        stats_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\n'
        stats_text += f'Bias = {bias:+.4f}\nVar Ratio = {var_ratio:.3f}'
        if n_total > max_points:
            stats_text += f'\n\nShowing {max_points:,}\nof {n_total:,} points'
        
        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
               family='monospace')
        
        ax.set_xlabel('MONC Truth', fontsize=13, fontweight='bold')
        ax.set_ylabel('Model Prediction', fontsize=13, fontweight='bold')
        ax.set_title(f'{model_name} - {var_label} (Full Domain Aggregate)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        output_file = output_dir / f'scatter_aggregate_{model_name}_{var}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    logger.info(f"    ✓ Saved aggregate scatter plots")


# ===================================================================
#  NON-ZERO EVALUATION
# ===================================================================

def calculate_nonzero_metrics(predictions: Dict, truth: Dict, threshold: float = 1e-10) -> Dict:
    """
    Calculate metrics for all data vs non-zero data on FULL DOMAIN.
    """
    results = {}
    
    for var in ['visc_coeff', 'diff_coeff']:
        if var not in predictions or var not in truth:
            continue
        
        pred_flat = predictions[var].flatten() if predictions[var].ndim > 1 else predictions[var]
        true_flat = truth[var].flatten() if truth[var].ndim > 1 else truth[var]
        
        valid = ~(np.isnan(pred_flat) | np.isnan(true_flat) | 
                 np.isinf(pred_flat) | np.isinf(true_flat))
        
        pred_valid = pred_flat[valid]
        true_valid = true_flat[valid]
        
        # All data metrics
        if len(pred_valid) > 0:
            r2_all = r2_score(true_valid, pred_valid)
            rmse_all = np.sqrt(mean_squared_error(true_valid, pred_valid))
            mae_all = mean_absolute_error(true_valid, pred_valid)
            bias_all = np.mean(pred_valid - true_valid)
        else:
            r2_all = rmse_all = mae_all = bias_all = np.nan
        
        # Non-zero data metrics
        nonzero_mask = np.abs(true_valid) > threshold
        true_nonzero = true_valid[nonzero_mask]
        pred_nonzero = pred_valid[nonzero_mask]
        
        zero_fraction = 1 - (np.sum(nonzero_mask) / len(true_valid))
        
        if len(true_nonzero) > 0:
            r2_nz = r2_score(true_nonzero, pred_nonzero)
            rmse_nz = np.sqrt(mean_squared_error(true_nonzero, pred_nonzero))
            mae_nz = mean_absolute_error(true_nonzero, pred_nonzero)
            bias_nz = np.mean(pred_nonzero - true_nonzero)
            relative_rmse = rmse_nz / np.mean(np.abs(true_nonzero))
            mean_truth_nz = np.mean(true_nonzero)
        else:
            r2_nz = rmse_nz = mae_nz = bias_nz = relative_rmse = mean_truth_nz = np.nan
        
        results[var] = {
            'all': {
                'r2': float(r2_all),
                'rmse': float(rmse_all),
                'mae': float(mae_all),
                'bias': float(bias_all),
                'n_valid': int(len(pred_valid))
            },
            'nonzero': {
                'r2': float(r2_nz),
                'rmse': float(rmse_nz),
                'mae': float(mae_nz),
                'bias': float(bias_nz),
                'n_valid': int(len(true_nonzero)),
                'relative_rmse': float(relative_rmse),
                'mean_truth': float(mean_truth_nz)
            },
            'zero_fraction': float(zero_fraction),
            'threshold': float(threshold)
        }
    
    return results


def plot_nonzero_comparison(all_nonzero_metrics: Dict, output_dir: Path):
    """
    Plot comparison of all-data vs non-zero metrics.
    """
    output_dir = Path(output_dir)
    
    logger.info("\nCreating non-zero comparison plots...")
    
    model_names = list(all_nonzero_metrics.keys())
    
    for var in ['visc_coeff', 'diff_coeff']:
        if not all(var in all_nonzero_metrics[m] for m in model_names):
            continue
        
        var_label = var.replace('_coeff', '').upper()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # R² comparison
        ax = axes[0]
        x_pos = np.arange(len(model_names))
        width = 0.35
        
        r2_all = [all_nonzero_metrics[m][var]['all']['r2'] for m in model_names]
        r2_nz = [all_nonzero_metrics[m][var]['nonzero']['r2'] for m in model_names]
        
        bars1 = ax.bar(x_pos - width/2, r2_all, width, label='All Data', 
                      alpha=0.8, color='#2E86AB')
        bars2 = ax.bar(x_pos + width/2, r2_nz, width, label='Non-Zero Only',
                      alpha=0.8, color='#E63946')
        
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel('R²', fontsize=11, fontweight='bold')
        ax.set_title(f'{var_label} - R² Comparison (Full Domain)', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('Baseline-', 'B-').replace('Ri-', 'Ri-') 
                           for m in model_names], fontsize=9, rotation=15, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # RMSE comparison
        ax = axes[1]
        rmse_all = [all_nonzero_metrics[m][var]['all']['rmse'] for m in model_names]
        rmse_nz = [all_nonzero_metrics[m][var]['nonzero']['rmse'] for m in model_names]
        
        bars1 = ax.bar(x_pos - width/2, rmse_all, width, label='All Data',
                      alpha=0.8, color='#2E86AB')
        bars2 = ax.bar(x_pos + width/2, rmse_nz, width, label='Non-Zero Only',
                      alpha=0.8, color='#E63946')
        
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel('RMSE', fontsize=11, fontweight='bold')
        ax.set_title(f'{var_label} - RMSE Comparison (Full Domain)', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('Baseline-', 'B-').replace('Ri-', 'Ri-') 
                           for m in model_names], fontsize=9, rotation=15, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_file = output_dir / f'nonzero_comparison_{var}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    logger.info(f"  ✓ Saved non-zero comparison plots")


# ===================================================================
#  CONSTRAINT ANALYSIS
# ===================================================================

def analyze_constraints(predictions_unc: Dict, predictions_con: Dict, 
                       truth: Dict, model_name: str) -> Dict:
    """
    Analyze impact of positivity constraints on FULL DOMAIN.
    """
    comparison = {}
    
    for var in ['visc_coeff', 'diff_coeff']:
        if var not in predictions_unc or var not in truth:
            continue
        
        pred_unc_flat = predictions_unc[var].flatten() if predictions_unc[var].ndim > 1 else predictions_unc[var]
        pred_con_flat = predictions_con[var].flatten() if predictions_con[var].ndim > 1 else predictions_con[var]
        true_flat = truth[var].flatten() if truth[var].ndim > 1 else truth[var]
        
        valid = ~(np.isnan(pred_unc_flat) | np.isnan(pred_con_flat) | np.isnan(true_flat) |
                 np.isinf(pred_unc_flat) | np.isinf(pred_con_flat) | np.isinf(true_flat))
        
        pred_unc_valid = pred_unc_flat[valid]
        pred_con_valid = pred_con_flat[valid]
        true_valid = true_flat[valid]
        
        # Metrics
        r2_unc = r2_score(true_valid, pred_unc_valid)
        r2_con = r2_score(true_valid, pred_con_valid)
        rmse_unc = np.sqrt(mean_squared_error(true_valid, pred_unc_valid))
        rmse_con = np.sqrt(mean_squared_error(true_valid, pred_con_valid))
        
        n_negative = np.sum(pred_unc_valid < 0)
        n_clipped = np.sum((pred_unc_valid < 0) & (pred_con_valid == 0))
        
        comparison[var] = {
            'unconstrained': {
                'r2': float(r2_unc),
                'rmse': float(rmse_unc),
                'bias': float(np.mean(pred_unc_valid - true_valid)),
                'n_negative': int(n_negative),
                'pct_negative': float(100 * n_negative / len(pred_unc_valid))
            },
            'constrained': {
                'r2': float(r2_con),
                'rmse': float(rmse_con),
                'bias': float(np.mean(pred_con_valid - true_valid)),
                'n_clipped': int(n_clipped)
            },
            'delta_r2': float(r2_con - r2_unc),
            'delta_rmse': float(rmse_con - rmse_unc)
        }
    
    return comparison


def plot_constraint_impact(all_constraint_analysis: Dict, output_dir: Path):
    """
    Plot constraint impact for all models.
    """
    output_dir = Path(output_dir)
    
    logger.info("\nCreating constraint impact plots...")
    
    model_names = list(all_constraint_analysis.keys())
    
    for var in ['visc_coeff', 'diff_coeff']:
        if not all(var in all_constraint_analysis[m] for m in model_names):
            continue
        
        var_label = var.replace('_coeff', '').upper()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Delta R²
        ax = axes[0]
        delta_r2 = [all_constraint_analysis[m][var]['delta_r2'] for m in model_names]
        
        colors = [get_model_color_and_style(m, model_names)[0] for m in model_names]
        bars = ax.bar(range(len(model_names)), delta_r2, alpha=0.8, color=colors)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:+.4f}', ha='center', 
                   va='bottom' if height > 0 else 'top', fontsize=9)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel('ΔR² (Constrained - Unconstrained)', fontsize=11, fontweight='bold')
        ax.set_title(f'{var_label} - R² Change (Full Domain)', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([m.replace('Baseline-', 'B-').replace('Ri-', 'Ri-') 
                           for m in model_names], fontsize=9, rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Negative values percentage
        ax = axes[1]
        pct_neg = [all_constraint_analysis[m][var]['unconstrained']['pct_negative'] 
                  for m in model_names]
        
        bars = ax.bar(range(len(model_names)), pct_neg, alpha=0.8, color=colors)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel('% Negative Predictions', fontsize=11, fontweight='bold')
        ax.set_title(f'{var_label} - Constraint Violations (Full Domain)', 
                    fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([m.replace('Baseline-', 'B-').replace('Ri-', 'Ri-') 
                           for m in model_names], fontsize=9, rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_file = output_dir / f'constraint_impact_{var}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    logger.info(f"  ✓ Saved constraint impact plots")


# ===================================================================
#  MAIN EXECUTION
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description='Comprehensive ML-SGS Analysis Suite')
    
    parser.add_argument('--mode', type=str, required=True, choices=['single', 'timeseries'])
    parser.add_argument('--nc-file', type=Path, help='NetCDF file (single mode)')
    parser.add_argument('--data-dir', type=Path, help='Directory with NetCDF files (timeseries mode)')
    parser.add_argument('--output', type=Path, required=True)
    
    # Model paths
    parser.add_argument('--baseline-mlp', type=Path, default=None)
    parser.add_argument('--baseline-resmlp', type=Path, default=None)
    parser.add_argument('--baseline-tabtransformer', type=Path, default=None)
    parser.add_argument('--ri-mlp', type=Path, default=None)
    parser.add_argument('--ri-resmlp', type=Path, default=None)
    parser.add_argument('--ri-tabtransformer', type=Path, default=None)
    
    parser.add_argument('--scaler-dir', type=Path, required=True)
    parser.add_argument('--time-idx', type=int, default=0, help='Time index within NetCDF file')
    parser.add_argument('--k-min', type=int, default=0, help='Minimum vertical level (FULL DOMAIN)')
    parser.add_argument('--k-max', type=int, default=219, help='Maximum vertical level (FULL DOMAIN)')
    parser.add_argument('--n-workers', type=int, default=None)
    parser.add_argument('--k-levels', type=int, nargs='+', default=[10, 50, 100, 150, 200],
                       help='K-levels for 2D slice VISUALIZATION only')
    
    args = parser.parse_args()
    
    # Build model paths
    baseline_paths = {}
    if args.baseline_mlp: baseline_paths['MLP'] = args.baseline_mlp
    if args.baseline_resmlp: baseline_paths['ResMLP'] = args.baseline_resmlp
    if args.baseline_tabtransformer: baseline_paths['TabTransformer'] = args.baseline_tabtransformer
    
    ri_paths = {}
    if args.ri_mlp: ri_paths['MLP'] = args.ri_mlp
    if args.ri_resmlp: ri_paths['ResMLP'] = args.ri_resmlp
    if args.ri_tabtransformer: ri_paths['TabTransformer'] = args.ri_tabtransformer
    
    if not baseline_paths and not ri_paths:
        parser.error("Must provide at least one model")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize engine
    logger.info(f"\n{'='*80}")
    logger.info("COMPREHENSIVE ML-SGS ANALYSIS SUITE")
    logger.info(f"{'='*80}\n")
    
    # Domain configuration
    domain_info = {
        'k_min': args.k_min,
        'k_max': args.k_max,
        'n_levels': args.k_max - args.k_min + 1
    }
    
    logger.info(f"DOMAIN CONFIGURATION:")
    logger.info(f"  Vertical domain: k={args.k_min} to k={args.k_max} ({domain_info['n_levels']} levels)")
    logger.info(f"  2D slice visualization: k={args.k_levels}")
    logger.info(f"  Time index within file: {args.time_idx}\n")
    
    logger.info("ANALYSIS SCOPE:")
    logger.info("  ✓ All metrics computed on FULL domain (k_min to k_max)")
    logger.info("  ✓ Spatial diagnostics use FULL domain")
    logger.info("  ✓ Vertical profiles use FULL domain")
    logger.info("  ✓ Time series metrics use FULL domain")
    logger.info("  ✓ Timestamps extracted as simulation time (seconds) from filenames")
    logger.info("  ✓ 2D slices visualized at selected k-levels only\n")
    
    engine = UnifiedInferenceEngine(
        baseline_paths=baseline_paths if baseline_paths else None,
        ri_paths=ri_paths if ri_paths else None,
        scaler_dir=args.scaler_dir,
        n_workers=args.n_workers
    )
    
    model_names = list(engine.models.keys())
    logger.info(f"Models loaded: {', '.join(model_names)}\n")
    
    # ========== SINGLE MODE ==========
    if args.mode == 'single':
        logger.info(f"Mode: Single file analysis")
        logger.info(f"File: {args.nc_file}\n")
        
        # Run inference
        predictions = engine.predict_3d_domain(args.nc_file, args.time_idx, args.k_min, args.k_max)
        truth = extract_truth_from_netcdf(args.nc_file, args.time_idx, args.k_min, args.k_max)
        
        # Storage
        all_spatial = {}
        all_profiles = {}
        all_nonzero = {}
        all_constraint = {}
        
        for model_name in model_names:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {model_name}")
            logger.info(f"{'='*60}")
            
            pred = predictions[model_name]
            
            # 1. Spatial diagnostics
            spatial = diagnose_spatial_patterns(pred, truth, output_dir / 'spatial', 
                                               model_name, time_idx=args.time_idx)
            all_spatial[model_name] = spatial
            
            # 2. Vertical profiles
            logger.info(f"\nComputing vertical profiles (FULL DOMAIN)...")
            profiles = compute_vertical_profiles(pred, truth, model_name)
            all_profiles[model_name] = profiles
            
            with open(output_dir / f'vertical_profiles_{model_name}.json', 'w') as f:
                json.dump(profiles, f, indent=2)
            logger.info(f"✓ Saved: vertical_profiles_{model_name}.json")
            
            # 3. 2D slices (visualization only)
            plot_2d_slices(pred, truth, output_dir, model_name, args.k_levels)
            
            # 4. Scatter plots (full domain)
            plot_aggregate_scatter_density(pred, truth, output_dir, model_name)
            
            # 5. Non-zero metrics (full domain)
            logger.info(f"\nComputing non-zero metrics (FULL DOMAIN)...")
            nonzero = calculate_nonzero_metrics(pred, truth)
            all_nonzero[model_name] = nonzero
            
            with open(output_dir / f'nonzero_metrics_{model_name}.json', 'w') as f:
                json.dump(nonzero, f, indent=2, cls=NumpyEncoder)
            logger.info(f"✓ Saved: nonzero_metrics_{model_name}.json")
            
            # 6. Constraint analysis (full domain)
            logger.info(f"\nAnalyzing constraints (FULL DOMAIN)...")
            pred_con = apply_physical_constraints(pred, verbose=True)
            constraint = analyze_constraints(pred, pred_con, truth, model_name)
            all_constraint[model_name] = constraint
            
            with open(output_dir / f'constraint_analysis_{model_name}.json', 'w') as f:
                json.dump(constraint, f, indent=2, cls=NumpyEncoder)
            logger.info(f"✓ Saved: constraint_analysis_{model_name}.json")
        
        # Cross-model plots
        logger.info(f"\n{'='*60}")
        logger.info("CREATING CROSS-MODEL COMPARISONS")
        logger.info(f"{'='*60}")
        
        plot_spatial_diagnostics(all_spatial, output_dir / 'spatial')
        plot_vertical_profiles(all_profiles, output_dir, domain_info)
        plot_nonzero_comparison(all_nonzero, output_dir)
        plot_constraint_impact(all_constraint, output_dir)
        
        # Summary report
        summary = {
            'analysis_mode': 'single',
            'domain_config': domain_info,
            'file': str(args.nc_file),
            'time_idx': args.time_idx,
            'models_analyzed': model_names,
            'visualization_k_levels': args.k_levels
        }
        
        with open(output_dir / 'analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n✅ Single file analysis complete: {output_dir}\n")
    
    # ========== TIMESERIES MODE ==========
    elif args.mode == 'timeseries':
        nc_files = sorted(
            args.data_dir.glob('*.nc'),
            key=lambda p: int(re.search(r'(\d+)\.nc$', p.name).group(1)) if re.search(r'(\d+)\.nc$', p.name) else 0
        )
        
        logger.info(f"Mode: Time series analysis")
        logger.info(f"Files: {len(nc_files)} files\n")
        
        # Extract timestamps (simulation time in seconds)
        timestamps, use_real_time = get_timestamps_from_files(nc_files)
        
        # Storage
        metrics_timeseries = {m: [] for m in model_names}
        prediction_timeseries = {m: [] for m in model_names}
        truth_timeseries = {'visc_coeff': [], 'diff_coeff': []}
        
        # Aggregate storage (FULL DOMAIN)
        agg_pred = {m: {'visc_coeff': [], 'diff_coeff': []} for m in model_names}
        agg_truth = {'visc_coeff': [], 'diff_coeff': []}
        
        # Process each file
        logger.info(f"\nProcessing time series (FULL DOMAIN: k={args.k_min} to {args.k_max})...")
        for t, nc_file in enumerate(nc_files):
            logger.info(f"  [{t+1}/{len(nc_files)}] {nc_file.name} | t={timestamps[t]:.0f}s")
            
            predictions = engine.predict_3d_domain(nc_file, args.time_idx, args.k_min, args.k_max)
            truth = extract_truth_from_netcdf(nc_file, args.time_idx, args.k_min, args.k_max)
            
            # Store time series
            for var in ['visc_coeff', 'diff_coeff']:
                truth_timeseries[var].append(truth[var])
                agg_truth[var].append(truth[var].flatten())
            
            for model_name in model_names:
                pred = predictions[model_name]
                
                # Metrics for this timestep
                metrics = compute_metrics_for_timestep(pred, truth)
                metrics_timeseries[model_name].append(metrics)
                
                # Store predictions
                prediction_timeseries[model_name].append(pred)
                
                # Aggregate
                for var in ['visc_coeff', 'diff_coeff']:
                    agg_pred[model_name][var].append(pred[var].flatten())
        
        # Concatenate aggregates
        logger.info("\nAggregating results from all timesteps...")
        for var in ['visc_coeff', 'diff_coeff']:
            agg_truth[var] = np.concatenate(agg_truth[var])
            logger.info(f"  {var}: {len(agg_truth[var]):,} total points")
            for model_name in model_names:
                agg_pred[model_name][var] = np.concatenate(agg_pred[model_name][var])
        
        # Save time series data
        logger.info("\nSaving time series data...")
        
        # Add timestamps to metrics
        metrics_with_time = {}
        for model_name in model_names:
            metrics_with_time[model_name] = []
            for t, metrics in enumerate(metrics_timeseries[model_name]):
                metrics_with_time[model_name].append({
                    'simulation_time_seconds': float(timestamps[t]),
                    'metrics': metrics
                })
        
        with open(output_dir / 'metrics_timeseries.json', 'w') as f:
            json.dump(metrics_with_time, f, indent=2, cls=NumpyEncoder)
        logger.info("✓ Saved: metrics_timeseries.json")
        
        # Plot time series
        logger.info(f"\n{'='*60}")
        logger.info("CREATING TIME SERIES PLOTS")
        logger.info(f"{'='*60}")
        
        plot_metrics_timeseries(metrics_timeseries, timestamps, output_dir, use_real_time)
        plot_prediction_timeseries(prediction_timeseries, truth_timeseries, timestamps, 
                                   output_dir, use_real_time=use_real_time)
        
        # Aggregate analysis
        logger.info(f"\n{'='*60}")
        logger.info("PERFORMING AGGREGATE ANALYSIS (ALL TIMESTEPS)")
        logger.info(f"{'='*60}")
        
        all_spatial = {}
        all_profiles = {}
        all_nonzero = {}
        all_constraint = {}
        
        for model_name in model_names:
            logger.info(f"\n{model_name} (aggregate analysis)...")
            
            # Use middle timestep for spatial/2D diagnostics (representative)
            mid_idx = len(nc_files) // 2
            pred_3d_mid = prediction_timeseries[model_name][mid_idx]
            truth_3d_mid = {
                'visc_coeff': truth_timeseries['visc_coeff'][mid_idx],
                'diff_coeff': truth_timeseries['diff_coeff'][mid_idx]
            }
            
            # Spatial diagnostics (representative timestep, full domain)
            spatial = diagnose_spatial_patterns(pred_3d_mid, truth_3d_mid, 
                                               output_dir / 'spatial', 
                                               model_name, time_idx=mid_idx)
            all_spatial[model_name] = spatial
            
            # Vertical profiles (averaged over all timesteps, full domain)
            logger.info(f"  Computing vertical profiles (averaged over {len(nc_files)} timesteps)...")
            
            profiles_per_timestep = []
            for t in range(len(nc_files)):
                pred_t = prediction_timeseries[model_name][t]
                truth_t = {
                    'visc_coeff': truth_timeseries['visc_coeff'][t],
                    'diff_coeff': truth_timeseries['diff_coeff'][t]
                }
                profile_t = compute_vertical_profiles(pred_t, truth_t, model_name)
                profiles_per_timestep.append(profile_t)
            
            # Average profiles over time
            profiles = {}
            for var in ['visc_coeff', 'diff_coeff']:
                if var in profiles_per_timestep[0]:
                    profiles[var] = {
                        'prediction': np.mean([p[var]['prediction'] for p in profiles_per_timestep], axis=0).tolist(),
                        'truth': np.mean([p[var]['truth'] for p in profiles_per_timestep], axis=0).tolist(),
                        'error': np.mean([p[var]['error'] for p in profiles_per_timestep], axis=0).tolist(),
                        'relative_error': np.mean([p[var]['relative_error'] for p in profiles_per_timestep], axis=0).tolist(),
                        'n_timesteps': len(nc_files)
                    }
            
            all_profiles[model_name] = profiles
            
            with open(output_dir / f'vertical_profiles_aggregate_{model_name}.json', 'w') as f:
                json.dump(profiles, f, indent=2)
            logger.info(f"  ✓ Saved: vertical_profiles_aggregate_{model_name}.json")
            
            # 2D slices (visualization only, middle timestep)
            logger.info(f"  Creating 2D slices at k-levels: {args.k_levels}")
            plot_2d_slices(pred_3d_mid, truth_3d_mid, output_dir, model_name, args.k_levels)
            
            # Aggregate scatter (all timesteps, full domain)
            logger.info(f"  Creating aggregate scatter plots (all timesteps)...")
            plot_aggregate_scatter_density(agg_pred[model_name], agg_truth, output_dir, model_name)
            
            # Non-zero metrics (aggregate, full domain)
            logger.info(f"  Computing non-zero metrics (aggregate)...")
            nonzero = calculate_nonzero_metrics(agg_pred[model_name], agg_truth)
            all_nonzero[model_name] = nonzero
            
            with open(output_dir / f'nonzero_metrics_aggregate_{model_name}.json', 'w') as f:
                json.dump(nonzero, f, indent=2, cls=NumpyEncoder)
            logger.info(f"  ✓ Saved: nonzero_metrics_aggregate_{model_name}.json")
            
            # Constraint analysis (aggregate, full domain)
            logger.info(f"  Analyzing constraints (aggregate)...")
            pred_con = apply_physical_constraints(agg_pred[model_name], verbose=False)
            constraint = analyze_constraints(agg_pred[model_name], pred_con, agg_truth, model_name)
            all_constraint[model_name] = constraint
            
            with open(output_dir / f'constraint_analysis_aggregate_{model_name}.json', 'w') as f:
                json.dump(constraint, f, indent=2, cls=NumpyEncoder)
            logger.info(f"  ✓ Saved: constraint_analysis_aggregate_{model_name}.json")
        
        # Cross-model plots
        logger.info(f"\n{'='*60}")
        logger.info("CREATING CROSS-MODEL COMPARISONS")
        logger.info(f"{'='*60}")
        
        plot_spatial_diagnostics(all_spatial, output_dir / 'spatial')
        plot_vertical_profiles(all_profiles, output_dir, domain_info)
        plot_nonzero_comparison(all_nonzero, output_dir)
        plot_constraint_impact(all_constraint, output_dir)
        
        # Summary report
        summary = {
            'analysis_mode': 'timeseries',
            'domain_config': domain_info,
            'timeseries_config': {
                'n_files': len(nc_files),
                'data_dir': str(args.data_dir),
                'time_range_seconds': {
                    'start': float(timestamps[0]),
                    'end': float(timestamps[-1])
                },
                'timestamps_extracted': use_real_time
            },
            'time_idx': args.time_idx,
            'models_analyzed': model_names,
            'visualization_k_levels': args.k_levels,
            'total_points_analyzed': int(len(agg_truth['visc_coeff']))
        }
        
        with open(output_dir / 'analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n✅ Time series analysis complete: {output_dir}\n")


if __name__ == '__main__':
    main()
