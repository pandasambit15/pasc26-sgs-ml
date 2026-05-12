#!/usr/bin/env python3
"""
Complete Best Models Analysis Pipeline
======================================

Comprehensive inference and evaluation for task-weighted models:
- MLP, ResMLP, TabTransformer (54-feature models)
- Full metrics suite (domain-averaged, height-wise, spatial)
- Enhanced visualizations (scatter, distributions, profiles)
- Physics quantity extraction and correlation analysis
- Time series and single file modes
- No sampling - uses FULL datasets

Author: Based on compare_all_models_claude_v4.py + enhancements
Date: 2025
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
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# Import model architectures
from train_new_coeff import UnifiedSGSCoefficientNetwork
from train_resmlp import ResMLPNetwork
from train_tab_transformer import TabTransformerNetwork

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="talk")


# ===================================================================
#  UTILITY CLASSES
# ===================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ===================================================================
#  DATA LOADING (FROM CLAUDE_V4)
# ===================================================================

class FastDataLoader:
    """Fast data loader with pre-loaded arrays for 54-feature extraction."""
    
    def __init__(self, nc_file: Path, time_idx: int):
        ds = xr.open_dataset(nc_file).load()
        time_dims = [dim for dim in ds.dims if dim.startswith('time_series_')]
        self.time_dim = time_dims[0] if time_dims else 'time'
        
        self.nx, self.ny = ds.sizes['x'], ds.sizes['y']
        self.nz_zn, self.nz_z = ds.sizes['zn'], ds.sizes['z']
        
        # Load all variables
        self.zu = self._load_var(ds, 'zu', time_idx, 'zn')
        self.zv = self._load_var(ds, 'zv', time_idx, 'zn')
        self.zth = self._load_var(ds, 'zth', time_idx, 'zn')
        self.zq_vapour = self._load_var(ds, 'zq_vapour', time_idx, 'zn')
        self.zq_cloud = self._load_var(ds, 'zq_cloud_liquid_mass', time_idx, 'zn')
        self.zw = self._load_var(ds, 'zw', time_idx, 'z')
        
        # Load metadata
        self.heights = ds['zn'].values if 'zn' in ds.coords else np.arange(self.nz_zn) * 50.0
        self.dx = float(ds['x_resolution'].isel({self.time_dim: time_idx}).values) if 'x_resolution' in ds else 100.0
        self.dy = float(ds['y_resolution'].isel({self.time_dim: time_idx}).values) if 'y_resolution' in ds else 100.0
        self.thref = float(ds.attrs['thref']) if 'thref' in ds.attrs else 300.0
        
        ds.close()

    def _load_var(self, ds, var_name, time_idx, coord):
        """Load variable with fallback names."""
        for name in [var_name, var_name[1:] if var_name.startswith('z') else 'z' + var_name]:
            if name in ds:
                var = ds[name]
                if self.time_dim in var.dims and coord in var.dims:
                    return var.isel({self.time_dim: time_idx}).values
        
        shape = (self.nx, self.ny, self.nz_zn if coord == 'zn' else self.nz_z)
        logger.warning(f"Variable '{var_name}' not found. Using zeros.")
        return np.zeros(shape, dtype=np.float32)


class FastFeatureExtractor:
    """Extracts 54 features from pre-loaded data."""
    
    def __init__(self, data_loader: FastDataLoader):
        self.data = data_loader

    def extract_point_features(self, i: int, j: int, k: int) -> Optional[np.ndarray]:
        """Extract 54 features at a single grid point."""
        features = []
        d = self.data
        nx, ny, nz_zn, nz_z = d.nx, d.ny, d.nz_zn, d.nz_z
        
        # Local values (6 features)
        features.extend([
            d.zu[i, j, k], 
            d.zv[i, j, k], 
            d.zw[i, j, min(k, nz_z - 1)], 
            d.zth[i, j, k], 
            d.zq_vapour[i, j, k], 
            d.zq_cloud[i, j, k]
        ])
        
        # Neighbor indices
        i_p, i_m = (i + 1) % nx, (i - 1) % nx
        j_p, j_m = (j + 1) % ny, (j - 1) % ny
        k_p, k_m = min(k + 1, nz_zn - 1), max(k - 1, 0)
        
        # Spatial neighbors (30 features)
        for var in [d.zu, d.zv, d.zth, d.zq_vapour, d.zq_cloud]:
            features.extend([
                var[i_p, j, k], var[i_m, j, k],
                var[i, j_p, k], var[i, j_m, k],
                var[i, j, k_p], var[i, j, k_m]
            ])
        
        # W-field neighbors (6 features)
        k_z, k_pz, k_mz = min(k, nz_z - 1), min(k_p, nz_z - 1), min(k_m, nz_z - 1)
        features.extend([
            d.zw[i_p, j, k_z], d.zw[i_m, j, k_z],
            d.zw[i, j_p, k_z], d.zw[i, j_m, k_z],
            d.zw[i, j, k_pz], d.zw[i, j, k_mz]
        ])
        
        # Grid parameters (5 features)
        height = d.heights[k]
        dz = d.heights[k_p] - height if k < nz_zn - 1 else height - d.heights[k_m]
        features.extend([d.dx, d.dy, dz, height, d.thref])
        
        # Normalized position (7 features)
        z_max = d.heights[-1]
        features.extend([
            k / nz_zn, 
            height, 
            z_max - height,
            i / nx, 
            j / ny,
            min(i, nx - 1 - i) / nx,
            min(j, ny - 1 - j) / ny
        ])
        
        return np.array(features, dtype=np.float32) if len(features) == 54 else None


def extract_chunk_worker(args):
    """Worker function for 54-feature parallel processing."""
    data_loader, coords_list = args
    extractor = FastFeatureExtractor(data_loader)
    return [extractor.extract_point_features(i, j, k) for i, j, k in coords_list]


# ===================================================================
#  INFERENCE ENGINE - NEW MODELS ONLY
# ===================================================================

class BestModelsInferenceEngine:
    """Inference engine for 54-feature best models (MLP, ResMLP, TabTransformer)."""
    
    def __init__(self, model_paths: Dict[str, Path], scaler_dir: Path, 
                 n_workers: Optional[int] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_workers = n_workers or max(1, cpu_count() - 2)
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Workers: {self.n_workers}")
        
        # Model architectures
        self.model_architectures = {
            'MLP': UnifiedSGSCoefficientNetwork,
            'ResMLP': ResMLPNetwork,
            'TabTransformer': TabTransformerNetwork
        }
        
        # Load models
        self.models = {}
        logger.info("Loading best models (54 features)...")
        for name, path in model_paths.items():
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            model = self.model_architectures[name](n_features=54)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            self.models[name] = model
            logger.info(f"  ✓ Loaded {name}")

        # Load scalers
        logger.info(f"Loading scalers from {scaler_dir}...")
        self.feature_scaler = joblib.load(scaler_dir / 'feature_scaler.pkl')
        self.visc_scaler = joblib.load(scaler_dir / 'visc_scaler.pkl')
        self.diff_scaler = joblib.load(scaler_dir / 'diff_scaler.pkl')
        self.ri_scaler = joblib.load(scaler_dir / 'richardson_scaler.pkl')
        logger.info("  ✓ Scalers loaded")

    def predict_3d_domain(self, nc_file: Path, time_idx: int, 
                         k_min: int, k_max: int) -> Dict:
        """Run inference for all best models."""
        
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
#  TRUTH EXTRACTION
# ===================================================================

def extract_truth_from_netcdf(nc_file: Path, time_idx: int, 
                              k_min: int, k_max: int) -> Dict:
    """Extract ground truth from NetCDF."""
    
    with xr.open_dataset(nc_file) as ds:
        time_dim = [d for d in ds.dims if d.startswith('time_series_')][0]
        
        k_max_zn = min(k_max, ds.sizes['zn'] - 1)
        k_max_z = min(k_max, ds.sizes['z'] - 1)
        
        visc = ds['visc_coeff'].isel({time_dim: time_idx, 'z': slice(k_min, k_max_z + 1)}).values
        diff = ds['diff_coeff'].isel({time_dim: time_idx, 'z': slice(k_min, k_max_z + 1)}).values
        ri = ds['ri_smag'].isel({time_dim: time_idx, 'zn': slice(k_min, k_max_zn + 1)}).values
        
        regime = np.select([ri < 0, (ri >= 0) & (ri < 0.25)], [0, 1], default=2)
        
        return {
            'visc_coeff': visc, 
            'diff_coeff': diff, 
            'richardson': ri, 
            'regime': regime,
            'heights': ds['zn'].values[k_min:k_max_zn+1], 
            'k_min': k_min, 
            'k_max': k_max_zn
        }


def determine_simulation_stage(regime_truth_data: np.ndarray) -> str:
    """Determine dominant physical stage."""
    
    total_points = regime_truth_data.size
    if total_points == 0: 
        return "Unknown"
    
    counts = {
        "Unstable": np.count_nonzero(regime_truth_data == 0),
        "Stable": np.count_nonzero(regime_truth_data == 1),
        "Supercritical": np.count_nonzero(regime_truth_data == 2)
    }
    
    dominant_stage = max(counts, key=counts.get)
    dominant_percentage = (counts[dominant_stage] / total_points) * 100
    
    return f"{dominant_stage} ({dominant_percentage:.1f}%)"


# ===================================================================
#  PHYSICS QUANTITY EXTRACTION
# ===================================================================

def extract_physical_quantities(nc_file: Path, time_idx: int = 0) -> Dict:
    """
    Extract physical quantities for correlation analysis.
    """
    
    with xr.open_dataset(nc_file) as ds:
        time_dim = [d for d in ds.dims if d.startswith('time_series_')][0]
        
        physics = {}
        
        # Basic state variables
        if 'zu' in ds or 'u' in ds:
            u_var = 'zu' if 'zu' in ds else 'u'
            physics['u_mean'] = float(np.nanmean(ds[u_var].isel({time_dim: time_idx}).values))
            physics['u_std'] = float(np.nanstd(ds[u_var].isel({time_dim: time_idx}).values))
        
        if 'zv' in ds or 'v' in ds:
            v_var = 'zv' if 'zv' in ds else 'v'
            physics['v_mean'] = float(np.nanmean(ds[v_var].isel({time_dim: time_idx}).values))
            physics['v_std'] = float(np.nanstd(ds[v_var].isel({time_dim: time_idx}).values))
        
        if 'zw' in ds or 'w' in ds:
            w_var = 'zw' if 'zw' in ds else 'w'
            physics['w_mean'] = float(np.nanmean(ds[w_var].isel({time_dim: time_idx}).values))
            physics['w_std'] = float(np.nanstd(ds[w_var].isel({time_dim: time_idx}).values))
            physics['w_max'] = float(np.nanmax(ds[w_var].isel({time_dim: time_idx}).values))
        
        if 'zth' in ds or 'th' in ds:
            th_var = 'zth' if 'zth' in ds else 'th'
            physics['theta_mean'] = float(np.nanmean(ds[th_var].isel({time_dim: time_idx}).values))
            physics['theta_std'] = float(np.nanstd(ds[th_var].isel({time_dim: time_idx}).values))
        
        # Moisture
        if 'zq_vapour' in ds or 'q_vapour' in ds:
            q_var = 'zq_vapour' if 'zq_vapour' in ds else 'q_vapour'
            physics['qv_mean'] = float(np.nanmean(ds[q_var].isel({time_dim: time_idx}).values))
        
        if 'zq_cloud_liquid_mass' in ds:
            physics['qcl_mean'] = float(np.nanmean(ds['zq_cloud_liquid_mass'].isel({time_dim: time_idx}).values))
        
        # Turbulence metrics
        if 'u_mean' in physics and 'v_mean' in physics:
            physics['wind_speed'] = float(np.sqrt(physics['u_mean']**2 + physics['v_mean']**2))
        
        if 'u_std' in physics and 'v_std' in physics:
            physics['turbulent_intensity'] = float(np.sqrt(physics['u_std']**2 + physics['v_std']**2))
        
        # Richardson number statistics
        if 'ri_smag' in ds:
            ri_data = ds['ri_smag'].isel({time_dim: time_idx}).values
            physics['ri_mean'] = float(np.nanmean(ri_data))
            physics['ri_std'] = float(np.nanstd(ri_data))
            physics['ri_min'] = float(np.nanmin(ri_data))
            physics['ri_max'] = float(np.nanmax(ri_data))
            
            # Stability fractions
            total = np.sum(~np.isnan(ri_data))
            if total > 0:
                physics['frac_unstable'] = float(np.sum(ri_data < 0) / total)
                physics['frac_stable'] = float(np.sum((ri_data >= 0) & (ri_data < 0.25)) / total)
                physics['frac_supercritical'] = float(np.sum(ri_data >= 0.25) / total)
        
        return physics


# ===================================================================
#  METRICS CALCULATION
# ===================================================================

def calculate_3d_metrics(predictions: Dict, truth: Dict) -> Dict:
    """Calculate domain-averaged metrics with variance ratio."""
    
    metrics = {}
    
    for key in ['visc_coeff', 'diff_coeff', 'richardson']:
        if key not in predictions or key not in truth:
            continue
        
        pred_array = predictions[key]
        true_array = truth[key]
        
        # Handle dimension mismatch
        nk_pred = pred_array.shape[2] if len(pred_array.shape) > 2 else 1
        nk_true = true_array.shape[2] if len(true_array.shape) > 2 else 1
        min_nk = min(nk_pred, nk_true)
        
        # Flatten
        pred = pred_array[:, :, :min_nk].flatten() if len(pred_array.shape) > 2 else pred_array.flatten()
        true = true_array[:, :, :min_nk].flatten() if len(true_array.shape) > 2 else true_array.flatten()
        
        # Filter valid points
        valid_mask = ~(np.isnan(pred) | np.isnan(true) | np.isinf(pred) | np.isinf(true))
        pred_valid = pred[valid_mask]
        true_valid = true[valid_mask]
        
        if len(pred_valid) == 0:
            metrics[key] = {'r2_score': np.nan, 'rmse': np.nan, 'n_valid_points': 0}
            continue
        
        # Calculate metrics
        r2 = r2_score(true_valid, pred_valid)
        rmse = np.sqrt(mean_squared_error(true_valid, pred_valid))
        mae = np.mean(np.abs(pred_valid - true_valid))
        var_ratio = np.var(pred_valid) / np.var(true_valid) if np.var(true_valid) > 0 else 0
        
        # Bias
        bias = np.mean(pred_valid - true_valid)
        
        metrics[key] = {
            'r2_score': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'n_valid_points': int(len(pred_valid)),
            'variance_ratio': float(var_ratio),
            'bias': float(bias)
        }
        
        if var_ratio < 0.3:
            logger.warning(f"⚠️  {key}: Mean collapse detected (var_ratio={var_ratio:.3f})")
    
    # Regime classification
    if 'regime' in predictions and 'regime' in truth:
        pred_regime = predictions['regime']
        true_regime = truth['regime']
        
        nk_pred = pred_regime.shape[2] if len(pred_regime.shape) > 2 else 1
        nk_true = true_regime.shape[2] if len(true_regime.shape) > 2 else 1
        min_nk = min(nk_pred, nk_true)
        
        pred_flat = pred_regime[:, :, :min_nk].flatten() if len(pred_regime.shape) > 2 else pred_regime.flatten()
        true_flat = true_regime[:, :, :min_nk].flatten() if len(true_regime.shape) > 2 else true_regime.flatten()
        
        valid_mask = (pred_flat >= 0) & (true_flat >= 0)
        pred_valid = pred_flat[valid_mask]
        true_valid = true_flat[valid_mask]
        
        if len(pred_valid) > 0:
            accuracy = accuracy_score(true_valid, pred_valid)
            metrics['regime'] = {
                'accuracy': float(accuracy), 
                'n_valid_points': int(len(pred_valid))
            }
        else:
            metrics['regime'] = {'accuracy': np.nan}
    
    return metrics

# ===================================================================
#  NON-ZERO METRICS CALCULATION
# ===================================================================

def calculate_nonzero_metrics(predictions: Dict, truth: Dict) -> Dict:
    """
    Calculate metrics separately for non-zero truth values.

    This is important for turbulent coefficients which can be zero
    in stable/quiescent regimes. Non-zero metrics show model performance
    on actively turbulent conditions.

    Returns
    -------
    Dict with structure:
    {
        'visc_coeff': {
            'all': {...},      # Metrics on all data
            'nonzero': {...},  # Metrics on non-zero truth only
            'zero_fraction': float,
            'threshold': float
        },
        ...
    }
    """

    metrics = {}

    # Threshold for "non-zero" (accounting for numerical precision)
    ZERO_THRESHOLD = 1e-10

    for key in ['visc_coeff', 'diff_coeff', 'richardson']:
        if key not in predictions or key not in truth:
            continue

        pred_array = predictions[key]
        true_array = truth[key]

        # Handle dimension mismatch
        nk_pred = pred_array.shape[2] if len(pred_array.shape) > 2 else 1
        nk_true = true_array.shape[2] if len(true_array.shape) > 2 else 1
        min_nk = min(nk_pred, nk_true)

        # Flatten
        pred = pred_array[:, :, :min_nk].flatten() if len(pred_array.shape) > 2 else pred_array.flatten()
        true = true_array[:, :, :min_nk].flatten() if len(true_array.shape) > 2 else true_array.flatten()

        # Filter valid points
        valid_mask = ~(np.isnan(pred) | np.isnan(true) | np.isinf(pred) | np.isinf(true))
        pred_valid = pred[valid_mask]
        true_valid = true[valid_mask]

        if len(pred_valid) == 0:
            metrics[key] = {
                'all': {'r2': np.nan, 'rmse': np.nan, 'n': 0},
                'nonzero': {'r2': np.nan, 'rmse': np.nan, 'n': 0},
                'zero_fraction': np.nan,
                'threshold': ZERO_THRESHOLD
            }
            continue

        # ALL DATA metrics
        r2_all = r2_score(true_valid, pred_valid)
        rmse_all = np.sqrt(mean_squared_error(true_valid, pred_valid))
        mae_all = np.mean(np.abs(pred_valid - true_valid))
        bias_all = np.mean(pred_valid - true_valid)

        # NON-ZERO mask (based on truth)
        nonzero_mask = np.abs(true_valid) > ZERO_THRESHOLD
        n_nonzero = np.sum(nonzero_mask)
        n_total = len(true_valid)
        zero_fraction = 1.0 - (n_nonzero / n_total)

        if n_nonzero < 10:
            # Not enough non-zero points
            metrics[key] = {
                'all': {
                    'r2': float(r2_all),
                    'rmse': float(rmse_all),
                    'mae': float(mae_all),
                    'bias': float(bias_all),
                    'n_valid': int(n_total)
                },
                'nonzero': {
                    'r2': np.nan,
                    'rmse': np.nan,
                    'mae': np.nan,
                    'bias': np.nan,
                    'n_valid': int(n_nonzero)
                },
                'zero_fraction': float(zero_fraction),
                'threshold': float(ZERO_THRESHOLD)
            }
            continue

        # NON-ZERO metrics
        pred_nonzero = pred_valid[nonzero_mask]
        true_nonzero = true_valid[nonzero_mask]

        r2_nonzero = r2_score(true_nonzero, pred_nonzero)
        rmse_nonzero = np.sqrt(mean_squared_error(true_nonzero, pred_nonzero))
        mae_nonzero = np.mean(np.abs(pred_nonzero - true_nonzero))
        bias_nonzero = np.mean(pred_nonzero - true_nonzero)

        # Relative error (normalized by mean of truth)
        mean_true_nonzero = np.mean(np.abs(true_nonzero))
        relative_rmse_nonzero = rmse_nonzero / mean_true_nonzero if mean_true_nonzero > 0 else np.nan

        metrics[key] = {
            'all': {
                'r2': float(r2_all),
                'rmse': float(rmse_all),
                'mae': float(mae_all),
                'bias': float(bias_all),
                'n_valid': int(n_total)
            },
            'nonzero': {
                'r2': float(r2_nonzero),
                'rmse': float(rmse_nonzero),
                'mae': float(mae_nonzero),
                'bias': float(bias_nonzero),
                'n_valid': int(n_nonzero),
                'relative_rmse': float(relative_rmse_nonzero),
                'mean_truth': float(mean_true_nonzero)
            },
            'zero_fraction': float(zero_fraction),
            'threshold': float(ZERO_THRESHOLD),
            'nonzero_range': {
                'truth_min': float(np.min(true_nonzero)),
                'truth_max': float(np.max(true_nonzero)),
                'pred_min': float(np.min(pred_nonzero)),
                'pred_max': float(np.max(pred_nonzero))
            }
        }

    return metrics


def create_nonzero_comparison_table(all_nonzero_metrics: Dict, output_dir: Path):
    """Create comparison table for non-zero metrics."""

    output_dir = Path(output_dir)

    rows = []

    for model_name, nz_metrics in all_nonzero_metrics.items():
        for var in ['visc_coeff', 'diff_coeff']:
            if var not in nz_metrics:
                continue

            var_label = var.replace('_coeff', '').upper()

            # ALL DATA row
            all_metrics = nz_metrics[var]['all']
            rows.append({
                'Model': model_name,
                'Variable': var_label,
                'Subset': 'All Data',
                'R²': all_metrics['r2'],
                'RMSE': all_metrics['rmse'],
                'MAE': all_metrics['mae'],
                'Bias': all_metrics['bias'],
                'N_Points': all_metrics['n_valid'],
                'Zero_Fraction': nz_metrics[var]['zero_fraction']
            })

            # NON-ZERO row
            nz = nz_metrics[var]['nonzero']
            rows.append({
                'Model': model_name,
                'Variable': var_label,
                'Subset': 'Non-Zero Only',
                'R²': nz['r2'],
                'RMSE': nz['rmse'],
                'MAE': nz['mae'],
                'Bias': nz['bias'],
                'N_Points': nz['n_valid'],
                'Zero_Fraction': np.nan
            })

    df = pd.DataFrame(rows)

    # Save
    csv_file = output_dir / 'nonzero_metrics_comparison.csv'
    df.to_csv(csv_file, index=False, float_format='%.4f')
    logger.info(f"✓ Saved: {csv_file.name}")

    # Print summary
    logger.info(f"\n{'='*90}")
    logger.info("NON-ZERO METRICS SUMMARY")
    logger.info(f"{'='*90}\n")

    for model_name in sorted(all_nonzero_metrics.keys()):
        logger.info(f"{model_name}:")
        nz_metrics = all_nonzero_metrics[model_name]

        for var in ['visc_coeff', 'diff_coeff']:
            if var not in nz_metrics:
                continue

            var_label = var.replace('_coeff', '').upper()
            all_m = nz_metrics[var]['all']
            nz_m = nz_metrics[var]['nonzero']
            zero_frac = nz_metrics[var]['zero_fraction']

            logger.info(f"\n  {var_label}:")
            logger.info(f"    Zero fraction: {zero_frac*100:.2f}%")
            logger.info(f"    ALL DATA:      R²={all_m['r2']:7.4f}, RMSE={all_m['rmse']:7.4f}, N={all_m['n_valid']:,}")
            logger.info(f"    NON-ZERO ONLY: R²={nz_m['r2']:7.4f}, RMSE={nz_m['rmse']:7.4f}, N={nz_m['n_valid']:,}")

            # Performance difference
            delta_r2 = nz_m['r2'] - all_m['r2']

            if delta_r2 > 0.05:
                logger.info(f"    ✓ Better on non-zero data (ΔR² = +{delta_r2:.4f})")
            elif delta_r2 < -0.05:
                logger.info(f"    ⚠ Worse on non-zero data (ΔR² = {delta_r2:.4f})")
            else:
                logger.info(f"    → Similar performance (ΔR² = {delta_r2:+.4f})")

    logger.info(f"\n{'='*90}\n")

    return df


def plot_nonzero_comparison(all_nonzero_metrics: Dict, output_dir: Path):
    """Plot comparison of all-data vs non-zero metrics."""

    output_dir = Path(output_dir)

    logger.info("\nCreating non-zero comparison plots...")

    model_names = sorted(all_nonzero_metrics.keys())

    for var in ['visc_coeff', 'diff_coeff']:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        var_label = var.replace('_coeff', '').title()

        # R² comparison
        ax = axes[0]
        x_pos = np.arange(len(model_names))
        width = 0.35

        r2_all = []
        r2_nonzero = []

        for model_name in model_names:
            if var in all_nonzero_metrics[model_name]:
                r2_all.append(all_nonzero_metrics[model_name][var]['all']['r2'])
                r2_nonzero.append(all_nonzero_metrics[model_name][var]['nonzero']['r2'])
            else:
                r2_all.append(0)
                r2_nonzero.append(0)

        bars1 = ax.bar(x_pos - width/2, r2_all, width,
                      label='All Data', alpha=0.8, color='#2E86AB')
        bars2 = ax.bar(x_pos + width/2, r2_nonzero, width,
                      label='Non-Zero Only', alpha=0.8, color='#E63946')

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        ax.axhline(y=0.7, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Target')
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel('R²', fontsize=11, fontweight='bold')
        ax.set_title(f'{var_label} - R² Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, fontsize=9, rotation=15, ha='right')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])

        # RMSE comparison
        ax = axes[1]

        rmse_all = []
        rmse_nonzero = []

        for model_name in model_names:
            if var in all_nonzero_metrics[model_name]:
                rmse_all.append(all_nonzero_metrics[model_name][var]['all']['rmse'])
                rmse_nonzero.append(all_nonzero_metrics[model_name][var]['nonzero']['rmse'])
            else:
                rmse_all.append(0)
                rmse_nonzero.append(0)

        bars1 = ax.bar(x_pos - width/2, rmse_all, width,
                      label='All Data', alpha=0.8, color='#2E86AB')
        bars2 = ax.bar(x_pos + width/2, rmse_nonzero, width,
                      label='Non-Zero Only', alpha=0.8, color='#E63946')

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel('RMSE', fontsize=11, fontweight='bold')
        ax.set_title(f'{var_label} - RMSE Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, fontsize=9, rotation=15, ha='right')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        output_file = output_dir / f'nonzero_comparison_{var}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"  ✓ Saved: {output_file.name}")


def plot_zero_fraction_analysis(all_nonzero_metrics: Dict, output_dir: Path):
    """Plot analysis of zero vs non-zero data distribution."""

    output_dir = Path(output_dir)

    logger.info("\nCreating zero-fraction analysis plot...")

    model_names = sorted(all_nonzero_metrics.keys())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, var in enumerate(['visc_coeff', 'diff_coeff']):
        ax = axes[idx]

        var_label = var.replace('_coeff', '').title()

        # Extract zero fractions and counts
        zero_fractions = []
        n_total = []
        n_nonzero = []

        for model_name in model_names:
            if var in all_nonzero_metrics[model_name]:
                zero_frac = all_nonzero_metrics[model_name][var]['zero_fraction']
                n_tot = all_nonzero_metrics[model_name][var]['all']['n_valid']
                n_nz = all_nonzero_metrics[model_name][var]['nonzero']['n_valid']

                zero_fractions.append(zero_frac * 100)
                n_total.append(n_tot)
                n_nonzero.append(n_nz)
            else:
                zero_fractions.append(0)
                n_total.append(0)
                n_nonzero.append(0)

        # Stacked bar chart
        n_zero = [n_total[i] - n_nonzero[i] for i in range(len(n_total))]

        x_pos = np.arange(len(model_names))

        bars1 = ax.bar(x_pos, n_nonzero, label='Non-Zero', alpha=0.8, color='#06A77D')
        bars2 = ax.bar(x_pos, n_zero, bottom=n_nonzero, label='Zero/Near-Zero', alpha=0.8, color='#E63946')

        # Add percentage labels
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            total_height = bar1.get_height() + bar2.get_height()

            if total_height == 0:
                continue

            # Non-zero percentage
            if bar1.get_height() > 0:
                pct_nonzero = (bar1.get_height() / total_height) * 100
                ax.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height()/2,
                       f'{pct_nonzero:.1f}%', ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white')

            # Zero percentage
            if bar2.get_height() > 0:
                pct_zero = (bar2.get_height() / total_height) * 100
                ax.text(bar2.get_x() + bar2.get_width()/2.,
                       bar1.get_height() + bar2.get_height()/2,
                       f'{pct_zero:.1f}%', ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white')

        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Points', fontsize=11, fontweight='bold')
        ax.set_title(f'{var_label} - Data Distribution', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, fontsize=9, rotation=15, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    plt.tight_layout()

    output_file = output_dir / 'zero_fraction_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"  ✓ Saved: {output_file.name}")

def calculate_heightwise_metrics(predictions: Dict, truth: Dict) -> Dict:
    """Calculate R² and RMSE at each vertical level."""
    
    logger.info("Calculating height-wise metrics...")
    
    heightwise_metrics = {}
    
    for model_name in ['MLP', 'ResMLP', 'TabTransformer']:
        if model_name not in predictions:
            continue
        
        heightwise_metrics[model_name] = {}
        
        for var_name in ['visc_coeff', 'diff_coeff', 'richardson']:
            if var_name not in predictions[model_name] or var_name not in truth:
                continue
            
            pred = predictions[model_name][var_name]
            true = truth[var_name]
            
            nk = min(pred.shape[2], true.shape[2])
            
            r2_by_height = []
            rmse_by_height = []
            n_valid_by_height = []
            
            for k in range(nk):
                pred_slice = pred[:, :, k].flatten()
                true_slice = true[:, :, k].flatten()
                
                valid = ~(np.isnan(pred_slice) | np.isnan(true_slice) | 
                         np.isinf(pred_slice) | np.isinf(true_slice))
                
                pred_valid = pred_slice[valid]
                true_valid = true_slice[valid]
                
                if len(pred_valid) < 10:
                    r2_by_height.append(np.nan)
                    rmse_by_height.append(np.nan)
                    n_valid_by_height.append(0)
                    continue
                
                try:
                    r2 = r2_score(true_valid, pred_valid)
                    rmse = np.sqrt(mean_squared_error(true_valid, pred_valid))
                    
                    r2_by_height.append(r2)
                    rmse_by_height.append(rmse)
                    n_valid_by_height.append(len(pred_valid))
                except:
                    r2_by_height.append(np.nan)
                    rmse_by_height.append(np.nan)
                    n_valid_by_height.append(len(pred_valid))
            
            heightwise_metrics[model_name][var_name] = {
                'r2': r2_by_height,
                'rmse': rmse_by_height,
                'n_valid': n_valid_by_height
            }
    
    return heightwise_metrics


def diagnose_spatial_patterns(predictions: Dict, truth: Dict, output_dir: Path):
    """
    Comprehensive spatial pattern analysis.
    Uses all heights for robust statistics.
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    logger.info(f"\n{'='*80}")
    logger.info("SPATIAL PATTERN DIAGNOSTICS")
    logger.info(f"{'='*80}")
    
    for var_name in ['visc_coeff', 'diff_coeff']:
        logger.info(f"\n{var_name.upper()}:")
        logger.info("-" * 60)
        
        results[var_name] = {}
        
        for model_name in ['MLP', 'ResMLP', 'TabTransformer']:
            if model_name not in predictions:
                continue
            
            pred = predictions[model_name][var_name]
            true = truth[var_name]
            
            # Use ALL height levels
            nk = min(pred.shape[2], true.shape[2])
            
            model_results = {
                'mean_match': [],
                'std_match': [],
                'r2_by_height': [],
                'spatial_autocorr_pred': [],
                'spatial_autocorr_true': []
            }
            
            for k in range(nk):
                pred_slice = pred[:, :, k]
                true_slice = true[:, :, k]
                
                valid = ~(np.isnan(pred_slice) | np.isnan(true_slice))
                if not np.any(valid):
                    continue
                
                pred_flat = pred_slice[valid]
                true_flat = true_slice[valid]
                
                if len(pred_flat) < 10:
                    continue
                
                # Statistics match
                mean_pred, mean_true = pred_flat.mean(), true_flat.mean()
                std_pred, std_true = pred_flat.std(), true_flat.std()
                
                mean_ratio = mean_pred / mean_true if mean_true != 0 else 0
                std_ratio = std_pred / std_true if std_true != 0 else 0
                
                model_results['mean_match'].append(mean_ratio)
                model_results['std_match'].append(std_ratio)
                
                # Point-wise R²
                try:
                    r2 = r2_score(true_flat, pred_flat)
                    model_results['r2_by_height'].append(r2)
                except:
                    pass
                
                # Spatial autocorrelation (sampled every 3rd level)
                if k % 3 == 0:
                    try:
                        if pred_slice.shape[0] > 1 and pred_slice.shape[1] > 10:
                            pred_x_corr = []
                            true_x_corr = []
                            
                            for row in range(pred_slice.shape[0] - 1):
                                pred_row = pred_slice[row, :]
                                pred_next = pred_slice[row + 1, :]
                                true_row = true_slice[row, :]
                                true_next = true_slice[row + 1, :]
                                
                                valid_corr = ~(np.isnan(pred_row) | np.isnan(pred_next) | 
                                             np.isnan(true_row) | np.isnan(true_next))
                                
                                if np.sum(valid_corr) > 10:
                                    try:
                                        pred_vals = pred_row[valid_corr]
                                        pred_next_vals = pred_next[valid_corr]
                                        
                                        if np.std(pred_vals) > 1e-10 and np.std(pred_next_vals) > 1e-10:
                                            pred_corr = np.corrcoef(pred_vals, pred_next_vals)[0, 1]
                                            if not np.isnan(pred_corr):
                                                pred_x_corr.append(pred_corr)
                                        
                                        true_vals = true_row[valid_corr]
                                        true_next_vals = true_next[valid_corr]
                                        
                                        if np.std(true_vals) > 1e-10 and np.std(true_next_vals) > 1e-10:
                                            true_corr = np.corrcoef(true_vals, true_next_vals)[0, 1]
                                            if not np.isnan(true_corr):
                                                true_x_corr.append(true_corr)
                                    except:
                                        pass
                            
                            if pred_x_corr:
                                model_results['spatial_autocorr_pred'].append(np.mean(pred_x_corr))
                            if true_x_corr:
                                model_results['spatial_autocorr_true'].append(np.mean(true_x_corr))
                    except:
                        pass
            
            # Aggregate results
            results[var_name][model_name] = {
                'mean_match_avg': float(np.mean(model_results['mean_match'])) if model_results['mean_match'] else 0,
                'std_match_avg': float(np.mean(model_results['std_match'])) if model_results['std_match'] else 0,
                'r2_avg': float(np.mean(model_results['r2_by_height'])) if model_results['r2_by_height'] else 0,
                'spatial_autocorr_pred_avg': float(np.mean(model_results['spatial_autocorr_pred'])) if model_results['spatial_autocorr_pred'] else np.nan,
                'spatial_autocorr_true_avg': float(np.mean(model_results['spatial_autocorr_true'])) if model_results['spatial_autocorr_true'] else np.nan
            }
            
            # Print diagnostics
            r = results[var_name][model_name]
            logger.info(f"\n  {model_name}:")
            logger.info(f"    Mean match:              {r['mean_match_avg']:.3f} "
                       f"({'✓' if abs(r['mean_match_avg'] - 1.0) < 0.1 else '✗'})")
            logger.info(f"    Std match:               {r['std_match_avg']:.3f} "
                       f"({'✓' if abs(r['std_match_avg'] - 1.0) < 0.2 else '✗'})")
            logger.info(f"    R² (avg):                {r['r2_avg']:.3f}")
            logger.info(f"    Spatial autocorr (pred): {r['spatial_autocorr_pred_avg']:.3f}")
            logger.info(f"    Spatial autocorr (true): {r['spatial_autocorr_true_avg']:.3f}")
            
            # Diagnosis
            if not np.isnan(r['spatial_autocorr_pred_avg']) and not np.isnan(r['spatial_autocorr_true_avg']):
                spatial_corr_loss = r['spatial_autocorr_true_avg'] - r['spatial_autocorr_pred_avg']
                
                if spatial_corr_loss > 0.3:
                    logger.info(f"    ⚠️  SPATIAL STRUCTURE LOST (Δcorr = {spatial_corr_loss:.3f})")
                elif spatial_corr_loss > 0.1:
                    logger.info(f"    ⚠️  Moderate spatial structure loss (Δcorr = {spatial_corr_loss:.3f})")
                else:
                    logger.info(f"    ✓ Spatial structure preserved (Δcorr = {spatial_corr_loss:.3f})")
    
    # Save results
    with open(output_dir / 'spatial_diagnostics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    
    return results


# ===================================================================
#  VISUALIZATION: SCATTER WITH DENSITY
# ===================================================================
def plot_scatter_with_density(aggregated_preds: Dict, aggregated_truth: Dict,
                              output_dir: Path, dataset_name: str = "aggregate"):
    """Clean scatter plots with proper spatial alignment."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    colors = {'MLP': '#2E86AB', 'ResMLP': '#06A77D', 'TabTransformer': '#A23B72'}
    
    for var_key in ['visc_coeff', 'diff_coeff']:
        logger.info(f"  {var_key}...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        var_label = var_key.replace('_coeff', ' Coefficient').title()
        
        for ax, model_name in zip(axes, ['MLP', 'ResMLP', 'TabTransformer']):
            pred_flat = aggregated_preds[model_name][var_key]
            true_flat = aggregated_truth[var_key]
            
            # CRITICAL: Ensure same length before comparison
            if len(pred_flat) != len(true_flat):
                logger.warning(f"    Length mismatch: pred={len(pred_flat)}, truth={len(true_flat)}")
                min_len = min(len(pred_flat), len(true_flat))
                pred_flat = pred_flat[:min_len]
                true_flat = true_flat[:min_len]
            
            # Filter valid + remove extreme outliers
            valid = ~(np.isnan(pred_flat) | np.isnan(true_flat) | 
                     np.isinf(pred_flat) | np.isinf(true_flat))
            
            if np.any(valid):
                # Remove points >1000x median (likely artifacts)
                pred_median = np.median(pred_flat[valid])
                true_median = np.median(true_flat[valid])
                extreme = (np.abs(pred_flat) < abs(pred_median) * 1000) & \
                         (np.abs(true_flat) < abs(true_median) * 1000)
                valid = valid & extreme
            
            pred_valid = pred_flat[valid]
            true_valid = true_flat[valid]
            
            if len(pred_valid) == 0:
                continue
            
            # Plot range from percentiles
            data_min = np.percentile(true_valid, 1)
            data_max = np.percentile(true_valid, 99)
            
            # Hexbin
            hexbin = ax.hexbin(true_valid, pred_valid, gridsize=60, cmap='viridis',
                             norm=LogNorm(vmin=1, vmax=max(10, len(pred_valid)/100)),
                             mincnt=1, alpha=0.9, edgecolors='none')
            
            cbar = plt.colorbar(hexbin, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Count (log)', fontsize=9, fontweight='bold')
            
            # 1:1 line
            ax.plot([data_min, data_max], [data_min, data_max], 
                   'r--', linewidth=2.5, alpha=0.8, label='1:1', zorder=10)
            
            # Metrics
            r2 = r2_score(true_valid, pred_valid)
            rmse = np.sqrt(mean_squared_error(true_valid, pred_valid))
            
            ax.text(0.03, 0.97, f'N = {len(pred_valid):,}\nR² = {r2:.4f}\nRMSE = {rmse:.4f}',
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                   family='monospace')
            
            ax.set_xlabel('MONC Truth', fontsize=10, fontweight='bold')
            ax.set_ylabel('Prediction', fontsize=10, fontweight='bold')
            ax.set_title(model_name, fontsize=11, fontweight='bold')
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim([data_min, data_max])
            ax.set_ylim([data_min, data_max])
        
        fig.suptitle(f'{var_label} - Model Comparison', fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        for ext in ['png', 'pdf']:
            output_file = output_dir / f'scatter_{var_key}_{dataset_name}.{ext}'
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


def plot_scatter_with_density_old(aggregated_preds: Dict, aggregated_truth: Dict,
                              output_dir: Path, dataset_name: str = "aggregate"):
    """
    Clean scatter plots with density - FULL DATA.
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    variables = ['visc_coeff', 'diff_coeff']
    model_names = ['MLP', 'ResMLP', 'TabTransformer']
    
    logger.info(f"\nCreating scatter plots: {dataset_name}")
    
    for var_key in variables:
        logger.info(f"  {var_key}...")
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
        
        for ax, model_name in zip(axes, model_names):
            pred_flat = aggregated_preds[model_name][var_key]
            true_flat = aggregated_truth[var_key]
            
            # Align and filter
            min_len = min(len(pred_flat), len(true_flat))
            pred_flat = pred_flat[:min_len]
            true_flat = true_flat[:min_len]
            
            valid = ~(np.isnan(pred_flat) | np.isnan(true_flat) | 
                     np.isinf(pred_flat) | np.isinf(true_flat))
            pred_valid = pred_flat[valid]
            true_valid = true_flat[valid]
            
            if len(pred_valid) == 0:
                continue
            
            # Data range
            data_min = np.percentile(true_valid, 0.5)
            data_max = np.percentile(true_valid, 99.5)
            
            # Hexbin
            hexbin = ax.hexbin(
                true_valid, pred_valid,
                gridsize=70,
                cmap='viridis',
                norm=LogNorm(vmin=1, vmax=max(10, len(pred_valid)/100)),
                mincnt=1,
                alpha=0.9,
                edgecolors='none'
            )
            
            # Colorbar
            cbar = plt.colorbar(hexbin, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Count (log)', fontsize=10, fontweight='bold')
            
            # 1:1 line
            ax.plot([data_min, data_max], [data_min, data_max], 
                   'r--', linewidth=2.5, alpha=0.8, label='1:1', zorder=10)
            
            # Metrics
            r2 = r2_score(true_valid, pred_valid)
            rmse = np.sqrt(mean_squared_error(true_valid, pred_valid))
            
            stats_text = f'N = {len(pred_valid):,}\nR² = {r2:.4f}\nRMSE = {rmse:.2f}'
            
            ax.text(0.03, 0.97, stats_text,
                   transform=ax.transAxes,
                   fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                   family='monospace')
            
            ax.set_xlabel('Truth (MONC)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Prediction', fontsize=12, fontweight='bold')
            ax.set_title(model_name, fontsize=14, fontweight='bold')
            ax.legend(loc='lower right', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
        
        fig.suptitle(f'{var_key.replace("_", " ").title()} - Scatter Comparison',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        output_file = output_dir / f'scatter_{var_key}_{dataset_name}.png'
        plt.savefig(output_file, dpi=250, bbox_inches='tight')
        plt.close()
        
        logger.info(f"    Saved: {output_file.name}")


# ===================================================================
#  VISUALIZATION: DISTRIBUTIONS
# ===================================================================

def plot_distributions(aggregated_preds: Dict, aggregated_truth: Dict,
                      output_dir: Path, dataset_name: str = "aggregate"):
    """
    Distribution comparison - histogram + KDE.
    """
    
    output_dir = Path(output_dir)
    
    colors = {
        'MLP': '#2E86AB',
        'ResMLP': '#06A77D',
        'TabTransformer': '#A23B72'
    }
    
    logger.info(f"\nCreating distribution plots: {dataset_name}")
    
    for var_key in ['visc_coeff', 'diff_coeff']:
        logger.info(f"  {var_key}...")
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Histogram
        ax = axes[0]
        true_flat = aggregated_truth[var_key]
        valid = ~(np.isnan(true_flat) | np.isinf(true_flat))
        true_valid = true_flat[valid]
        
        ax.hist(true_valid, bins=150, alpha=0.45, color='#C41E3A', 
               label=f'Truth (n={len(true_valid):,})', 
               density=True, edgecolor='none')
        
        for model_name, color in colors.items():
            pred_flat = aggregated_preds[model_name][var_key]
            valid = ~(np.isnan(pred_flat) | np.isinf(pred_flat))
            pred_valid = pred_flat[valid]
            
            ax.hist(pred_valid, bins=150, alpha=0.35, color=color,
                   label=f'{model_name} (n={len(pred_valid):,})', 
                   density=True, edgecolor='none')
        
        ax.set_xlabel(var_key.replace('_', ' ').title(), fontsize=13, fontweight='bold')
        ax.set_ylabel('Density', fontsize=13, fontweight='bold')
        ax.set_title('Histogram', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        
        # KDE
        ax = axes[1]
        
        # Sample for KDE
        kde_sample_size = min(len(true_valid), 100000)
        true_sample = np.random.choice(true_valid, kde_sample_size, replace=False) if len(true_valid) > kde_sample_size else true_valid
        
        x_range = np.linspace(np.percentile(true_valid, 0.5), np.percentile(true_valid, 99.5), 500)
        
        try:
            kde_truth = gaussian_kde(true_sample, bw_method='scott')
            ax.plot(x_range, kde_truth(x_range), 
                   color='#C41E3A', linewidth=4, label='Truth', alpha=0.9)
        except:
            pass
        
        for model_name, color in colors.items():
            pred_flat = aggregated_preds[model_name][var_key]
            valid = ~(np.isnan(pred_flat) | np.isinf(pred_flat))
            pred_valid = pred_flat[valid]
            
            pred_sample = np.random.choice(pred_valid, kde_sample_size, replace=False) if len(pred_valid) > kde_sample_size else pred_valid
            
            try:
                kde_pred = gaussian_kde(pred_sample, bw_method='scott')
                ax.plot(x_range, kde_pred(x_range), 
                       color=color, linewidth=3, label=model_name, alpha=0.85)
            except:
                pass
        
        ax.set_xlabel(var_key.replace('_', ' ').title(), fontsize=13, fontweight='bold')
        ax.set_ylabel('Density', fontsize=13, fontweight='bold')
        ax.set_title('KDE', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'{var_key.replace("_", " ").title()} - Distribution Comparison',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        output_file = output_dir / f'distributions_{var_key}_{dataset_name}.png'
        plt.savefig(output_file, dpi=250, bbox_inches='tight')
        plt.close()
        
        logger.info(f"    Saved: {output_file.name}")


# ===================================================================
#  VISUALIZATION: VERTICAL PROFILES
# ===================================================================

def plot_vertical_profiles(predictions: Dict, truth: Dict, output_dir: Path):
    """
    Enhanced vertical profiles.
    """
    
    output_dir = Path(output_dir)
    
    logger.info("\nCreating vertical profiles...")
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    
    variables = [
        ('visc_coeff', 'Viscosity'), 
        ('diff_coeff', 'Diffusivity'), 
        ('richardson', 'Richardson')
    ]
    
    colors = {
        'MLP': '#2E86AB',
        'ResMLP': '#06A77D',
        'TabTransformer': '#A23B72'
    }
    
    heights = predictions['shared']['heights']
    
    for ax, (key, title) in zip(axes, variables):
        # Truth
        true_mean = np.nanmean(truth[key], axis=(0, 1))
        true_std = np.nanstd(truth[key], axis=(0, 1))
        
        ax.plot(true_mean, heights, 'r--', lw=4, 
               label='MONC Truth', zorder=100, alpha=0.9)
        ax.fill_betweenx(heights, true_mean - true_std, true_mean + true_std, 
                        color='r', alpha=0.2, label='Truth ±1σ')
        
        # Models
        for model_name, preds in predictions.items():
            if model_name == 'shared': 
                continue
            
            pred_mean = np.nanmean(preds[key], axis=(0, 1))
            pred_std = np.nanstd(preds[key], axis=(0, 1))
            
            ax.plot(pred_mean, heights, 
                   color=colors[model_name], lw=3, 
                   label=model_name, alpha=0.85)
            ax.fill_betweenx(heights, pred_mean - pred_std, pred_mean + pred_std, 
                            color=colors[model_name], alpha=0.1)
        
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_ylabel('Height (m)', fontsize=13, fontweight='bold')
        ax.set_xlabel(title, fontsize=13)
        ax.legend(loc='best', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'vertical_profiles.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved: {output_file.name}")


def plot_heightwise_metrics(heightwise_metrics: Dict, heights: np.ndarray, 
                            output_dir: Path):
    """Plot height-wise R² and RMSE."""
    
    output_dir = Path(output_dir)
    
    logger.info("\nCreating height-wise metric plots...")
    
    colors = {
        'MLP': '#2E86AB',
        'ResMLP': '#06A77D',
        'TabTransformer': '#A23B72'
    }
    
    for var_name in ['visc_coeff', 'diff_coeff']:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # R²
        ax = axes[0]
        for model_name, color in colors.items():
            if model_name in heightwise_metrics and var_name in heightwise_metrics[model_name]:
                r2_values = heightwise_metrics[model_name][var_name]['r2']
                ax.plot(r2_values, heights, color=color, linewidth=2.5, 
                       label=model_name, marker='o', markersize=3)
        
        ax.axvline(x=0.7, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Target (0.7)')
        ax.set_xlabel('R²', fontsize=13, fontweight='bold')
        ax.set_ylabel('Height (m)', fontsize=13, fontweight='bold')
        ax.set_title(f'{var_name.replace("_", " ").title()} - R² by Height', 
                    fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.5, 1.0])
        
        # RMSE
        ax = axes[1]
        for model_name, color in colors.items():
            if model_name in heightwise_metrics and var_name in heightwise_metrics[model_name]:
                rmse_values = heightwise_metrics[model_name][var_name]['rmse']
                ax.plot(rmse_values, heights, color=color, linewidth=2.5, 
                       label=model_name, marker='o', markersize=3)
        
        ax.set_xlabel('RMSE', fontsize=13, fontweight='bold')
        ax.set_ylabel('Height (m)', fontsize=13, fontweight='bold')
        ax.set_title(f'{var_name.replace("_", " ").title()} - RMSE by Height', 
                    fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = output_dir / f'heightwise_{var_name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved: {output_file.name}")


# ===================================================================
#  VISUALIZATION: TIME SERIES
# ===================================================================
def plot_time_series(series_data: List[Dict], output_dir: Path):
    """Publication-quality time series with timestamps and error bars."""
    
    output_dir = Path(output_dir)
    timestamps = np.array([d['timestamp'] for d in series_data])
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    colors = {'MLP': '#2E86AB', 'ResMLP': '#06A77D', 'TabTransformer': '#A23B72', 'Truth': '#C41E3A'}
    
    variables = [
        ('visc_coeff', 'Viscosity Coefficient', axes[0]),
        ('diff_coeff', 'Diffusivity Coefficient', axes[1])
    ]
    
    for var_key, var_label, ax in variables:
        # Truth with error band
        truth_means = np.array([d['truth']['mean'][var_key] for d in series_data])
        truth_stds = np.array([d['truth']['std'][var_key] for d in series_data])
        
        ax.plot(timestamps, truth_means, color=colors['Truth'], linewidth=2.5, 
               label='MONC Truth', marker='o', markersize=6, zorder=10, linestyle='--')
        ax.fill_between(timestamps, truth_means - truth_stds, truth_means + truth_stds,
                       color=colors['Truth'], alpha=0.15, label='Truth ±1σ', zorder=5)
        
        # Model predictions with error bars
        for model_name in ['MLP', 'ResMLP', 'TabTransformer']:
            means = np.array([d['predictions'][model_name]['mean'][var_key] for d in series_data])
            stds = np.array([d['predictions'][model_name]['std'][var_key] for d in series_data])
            
            ax.errorbar(timestamps, means, yerr=stds, color=colors[model_name], linewidth=2,
                       label=model_name, marker='s', markersize=5, capsize=4, 
                       capthick=1.5, alpha=0.85, zorder=8)
        
        ax.set_ylabel(var_label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9, framealpha=0.95, ncol=2)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', labelsize=9)
        ax.ticklabel_format(style='plain', axis='x')
        ax.grid(True, which='minor', alpha=0.15, linestyle=':')
        ax.minorticks_on()
    
    plt.tight_layout()
    
    for ext in ['png', 'pdf']:
        output_file = output_dir / f'timeseries_publication.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"✓ Saved publication time series")


def plot_time_series_old(series_data: List[Dict], output_dir: Path):
    """Plot time series evolution."""
    
    output_dir = Path(output_dir)
    
    logger.info("\nCreating time series plots...")
    
    timestamps = [d['timestamp'] for d in series_data]
    times_hr = (np.array(timestamps) - timestamps[0]) / 3600.0
    
    colors = {
        'MLP': '#2E86AB',
        'ResMLP': '#06A77D',
        'TabTransformer': '#A23B72'
    }
    
    for var_name in ['visc_coeff', 'diff_coeff']:
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        
        # Mean
        ax = axes[0]
        truth_means = [d['truth']['mean'][var_name] for d in series_data]
        ax.plot(times_hr, truth_means, 'r--', linewidth=3, label='Truth', marker='o', markersize=5)
        
        for model_name, color in colors.items():
            means = [d['predictions'][model_name]['mean'][var_name] for d in series_data]
            ax.plot(times_hr, means, color=color, linewidth=2.5, label=model_name, marker='s', markersize=4)
        
        ax.set_ylabel(f'{var_name.replace("_", " ").title()}\n(mean)', fontsize=12, fontweight='bold')
        ax.set_title('Mean Evolution', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Std
        ax = axes[1]
        truth_stds = [d['truth']['std'][var_name] for d in series_data]
        ax.plot(times_hr, truth_stds, 'r--', linewidth=3, label='Truth', marker='o', markersize=5)
        
        for model_name, color in colors.items():
            stds = [d['predictions'][model_name]['std'][var_name] for d in series_data]
            ax.plot(times_hr, stds, color=color, linewidth=2.5, label=model_name, marker='s', markersize=4)
        
        ax.set_ylabel(f'{var_name.replace("_", " ").title()}\n(std)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        ax.set_title('Std Evolution', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = output_dir / f'timeseries_{var_name}.png'
        plt.savefig(output_file, dpi=250, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved: {output_file.name}")


# ===================================================================
#  VISUALIZATION: PHYSICS CORRELATIONS
# ===================================================================

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
            ax.plot(times_hr, values, linewidth=2.5, marker='o', markersize=5)
            ax.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
            ax.set_ylabel(qty.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.set_title(qty.replace('_', ' ').title(), fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'physics_evolution.png'
    plt.savefig(output_file, dpi=250, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved: {output_file.name}")


# ===================================================================
#  MAIN FUNCTION
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Complete Best Models Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Mode
    parser.add_argument('--mode', type=str, required=True,
                       choices=['single', 'timeseries'],
                       help='Analysis mode')
    
    # Data paths
    parser.add_argument('--nc-file', type=Path,
                       help='NetCDF file (for single mode)')
    parser.add_argument('--data-dir', type=Path,
                       help='Directory with NetCDF files (for timeseries mode)')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory')
    
    # Model paths
    parser.add_argument('--mlp-weights', type=Path, required=True)
    parser.add_argument('--resmlp-weights', type=Path, required=True)
    parser.add_argument('--tabtransformer-weights', type=Path, required=True)
    parser.add_argument('--scaler-dir', type=Path, required=True,
                       help='Directory with 54-feature scalers')
    
    # Processing parameters
    parser.add_argument('--time-idx', type=int, default=0)
    parser.add_argument('--k-min', type=int, default=2)
    parser.add_argument('--k-max', type=int, default=30)
    parser.add_argument('--n-workers', type=int, default=None)
    
    # Features
    parser.add_argument('--extract-physics', action='store_true',
                       help='Extract physical quantities')
    
    args = parser.parse_args()
    
    # Validation
    if args.mode == 'single' and not args.nc_file:
        parser.error("--nc-file required for single mode")
    if args.mode == 'timeseries' and not args.data_dir:
        parser.error("--data-dir required for timeseries mode")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize engine
    logger.info(f"\n{'='*80}")
    logger.info("INITIALIZING INFERENCE ENGINE")
    logger.info(f"{'='*80}\n")
    
    model_paths = {
        'MLP': args.mlp_weights,
        'ResMLP': args.resmlp_weights,
        'TabTransformer': args.tabtransformer_weights
    }
    
    engine = BestModelsInferenceEngine(
        model_paths, args.scaler_dir, args.n_workers
    )
    
    logger.info("✓ Engine ready\n")
    
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
        
        # Stage
        stage = determine_simulation_stage(truth['regime'])
        logger.info(f"\nStage: {stage}")
        
        # Metrics
        logger.info(f"\n{'='*80}")
        logger.info("CALCULATING METRICS")
        logger.info(f"{'='*80}\n")
        
        all_metrics = {}
        for name, preds in predictions.items():
            if name == 'shared':
                continue
            all_metrics[name] = calculate_3d_metrics(preds, truth)
        
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=2, cls=NumpyEncoder)
        logger.info(f"✓ Saved: metrics.json")
        
        # Print summary
        for name, metrics in all_metrics.items():
            logger.info(f"\n{name}:")
            logger.info(f"  Visc R²: {metrics['visc_coeff']['r2_score']:.4f}")
            logger.info(f"  Diff R²: {metrics['diff_coeff']['r2_score']:.4f}")
            logger.info(f"  Rich R²: {metrics['richardson']['r2_score']:.4f}")
        
        # Height-wise
        heightwise = calculate_heightwise_metrics(predictions, truth)
        with open(output_dir / 'heightwise_metrics.json', 'w') as f:
            json.dump(heightwise, f, indent=2, cls=NumpyEncoder)
        logger.info(f"\n✓ Saved: heightwise_metrics.json")
        
        # Spatial
        spatial = diagnose_spatial_patterns(predictions, truth, output_dir)
        
        # Visualizations
        logger.info(f"\n{'='*80}")
        logger.info("CREATING VISUALIZATIONS")
        logger.info(f"{'='*80}")
        
        plot_vertical_profiles(predictions, truth, output_dir)
        plot_heightwise_metrics(heightwise, truth['heights'], output_dir)
        
        # Aggregate for scatter
        agg_preds = {}
        agg_truth = {}
        for var in ['visc_coeff', 'diff_coeff', 'richardson']:
            agg_truth[var] = truth[var].flatten()
            for name in ['MLP', 'ResMLP', 'TabTransformer']:
                if name not in agg_preds:
                    agg_preds[name] = {}
                agg_preds[name][var] = predictions[name][var].flatten()
        
        plot_scatter_with_density(agg_preds, agg_truth, output_dir, 'single')
        plot_distributions(agg_preds, agg_truth, output_dir, 'single')
        
        # Physics
        if args.extract_physics:
            physics = extract_physical_quantities(args.nc_file, args.time_idx)
            with open(output_dir / 'physics.json', 'w') as f:
                json.dump(physics, f, indent=2)
            logger.info(f"\n✓ Saved: physics.json")
        
        logger.info(f"\n{'='*80}")
        logger.info("✓ SINGLE MODE COMPLETE")
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
        all_metrics = []
        series_data = []
        physics_series = []
        
        agg_preds = {name: {'visc_coeff': [], 'diff_coeff': [], 'richardson': []} 
                     for name in ['MLP', 'ResMLP', 'TabTransformer']}
        agg_truth = {'visc_coeff': [], 'diff_coeff': [], 'richardson': []}
        
        last_predictions = None
        last_truth = None
        
        # Process files
        for idx, nc_file in enumerate(files):
            logger.info(f"{'='*70}")
            logger.info(f"File {idx+1}/{len(files)}: {nc_file.name}")
            logger.info(f"{'='*70}")
            
            timestamp = int(re.search(r'(\d+)\.nc$', nc_file.name).group(1))
            
            # Inference
            predictions = engine.predict_3d_domain(
                nc_file, args.time_idx, args.k_min, args.k_max
            )
            truth = extract_truth_from_netcdf(
                nc_file, args.time_idx, args.k_min, args.k_max
            )
            
            last_predictions = predictions
            last_truth = truth
            
            # Metrics
            timestep_metrics = {}
            for name, preds in predictions.items():
                if name == 'shared':
                    continue
                timestep_metrics[name] = calculate_3d_metrics(preds, truth)
            
            all_metrics.append({
                'file': nc_file.name,
                'timestamp': timestamp,
                'metrics': timestep_metrics
            })
            
            # Time series
            timestep_series = {
                'timestamp': timestamp,
                'predictions': {},
                'truth': {}
            }
            
            for name, preds in predictions.items():
                if name == 'shared':
                    continue
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
            for var in ['visc_coeff', 'diff_coeff', 'richardson']:
                agg_truth[var].append(truth[var].flatten())
                for name in ['MLP', 'ResMLP', 'TabTransformer']:
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
            for name in ['MLP', 'ResMLP', 'TabTransformer']:
                agg_preds[name][var] = np.concatenate(agg_preds[name][var])
        
        # Save
        logger.info(f"\n{'='*70}")
        logger.info("SAVING RESULTS")
        logger.info(f"{'='*70}")
        
        with open(output_dir / 'metrics_timeseries.json', 'w') as f:
            json.dump(all_metrics, f, indent=2, cls=NumpyEncoder)
        logger.info(f"✓ Saved: metrics_timeseries.json")
        
        # Aggregate metrics
        aggregate_metrics = {}
        for name in ['MLP', 'ResMLP', 'TabTransformer']:
            temp_pred = {k: agg_preds[name][k] for k in ['visc_coeff', 'diff_coeff', 'richardson']}
            aggregate_metrics[name] = calculate_3d_metrics(temp_pred, agg_truth)
        
        with open(output_dir / 'metrics_aggregate.json', 'w') as f:
            json.dump(aggregate_metrics, f, indent=2, cls=NumpyEncoder)
        logger.info(f"✓ Saved: metrics_aggregate.json")
        
        logger.info("\nAGGREGATE RESULTS:")
        for name, metrics in aggregate_metrics.items():
            logger.info(f"\n  {name}:")
            logger.info(f"    Visc R²: {metrics['visc_coeff']['r2_score']:.4f}")
            logger.info(f"    Diff R²: {metrics['diff_coeff']['r2_score']:.4f}")
            logger.info(f"    Rich R²: {metrics['richardson']['r2_score']:.4f}")
        
        if args.extract_physics:
            with open(output_dir / 'physics_timeseries.json', 'w') as f:
                json.dump(physics_series, f, indent=2, cls=NumpyEncoder)
            logger.info(f"\n✓ Saved: physics_timeseries.json")
        
        # Visualizations
        logger.info(f"\n{'='*70}")
        logger.info("CREATING VISUALIZATIONS")
        logger.info(f"{'='*70}")
        
        plot_time_series(series_data, output_dir)
        
        if last_predictions and last_truth:
            plot_vertical_profiles(last_predictions, last_truth, output_dir)
            heightwise = calculate_heightwise_metrics(last_predictions, last_truth)
            plot_heightwise_metrics(heightwise, last_truth['heights'], output_dir)
            diagnose_spatial_patterns(last_predictions, last_truth, output_dir)
        
        plot_scatter_with_density(agg_preds, agg_truth, output_dir, f'timeseries_{len(files)}files')
        plot_distributions(agg_preds, agg_truth, output_dir, f'timeseries_{len(files)}files')
        
        if args.extract_physics:
            plot_physics_correlations(physics_series, output_dir)
        
        logger.info(f"\n{'='*80}")
        logger.info("✓ TIME SERIES ANALYSIS COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"\nResults: {output_dir}")
        logger.info(f"Files: {len(files)}")
        logger.info(f"Total points: {len(agg_truth['visc_coeff']):,}")
        logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    main()
