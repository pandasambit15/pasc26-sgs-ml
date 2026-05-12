#!/usr/bin/env python3
"""
Constrained Inference Analysis with Ri-Conditioning Support - FIXED VERSION
===========================================================================

CRITICAL FIX: Applies physical constraints AFTER inverse scaling, not before.

Usage:
    python constrained_inference_with_ri_fixed.py \
        --mode single \
        --nc-file data.nc \
        --baseline-mlp mlp.pth --ri-mlp ri_mlp.pth \
        --scaler-dir scalers/ --output results/
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import logging
from typing import Dict, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import re
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.colors import LogNorm

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
    NumpyEncoder,
    calculate_nonzero_metrics,
    create_nonzero_comparison_table,
    plot_nonzero_comparison,
    plot_zero_fraction_analysis
)

# Import from unified engine (for model loading)
from run_models_comparison_with_ri_v2 import UnifiedInferenceEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="paper")


# ===================================================================
#  FIXED CONSTRAINT APPLICATION
# ===================================================================

def apply_physical_constraints(predictions: Dict, verbose: bool = True) -> Dict:
    """
    Apply physical constraints to predictions IN PHYSICAL SPACE.
    
    CRITICAL: This must be called AFTER inverse_transform, not before!
    
    Constraints:
    - visc_coeff >= 0
    - diff_coeff >= 0
    - richardson: no constraint (can be negative)
    
    Parameters
    ----------
    predictions : Dict
        Model predictions ALREADY IN PHYSICAL SPACE with keys 
        'visc_coeff', 'diff_coeff', 'richardson', 'regime'
    verbose : bool
        If True, log statistics about clipped values
    
    Returns
    -------
    Dict : Constrained predictions
    """
    
    constrained = {}
    
    for key, values in predictions.items():
        if key in ['visc_coeff', 'diff_coeff']:
            # Apply non-negativity constraint IN PHYSICAL SPACE
            constrained[key] = np.maximum(values, 0.0)
            
            # Log statistics
            if verbose:
                n_negative = np.sum(values < 0)
                if n_negative > 0:
                    pct_negative = 100 * n_negative / values.size
                    logger.info(f"    {key}: {n_negative:,} negative values ({pct_negative:.2f}%) clipped to 0")
                    logger.info(f"      Original range: [{np.min(values):.4f}, {np.max(values):.4f}]")
                    logger.info(f"      Constrained range: [{np.min(constrained[key]):.4f}, {np.max(constrained[key]):.4f}]")
                else:
                    logger.info(f"    {key}: No negative values (already physically valid)")
        else:
            # No constraint for richardson and regime
            constrained[key] = values.copy()
    
    return constrained


def calculate_metrics_comparison(predictions_unconstrained: Dict,
                                 predictions_constrained: Dict,
                                 truth: Dict) -> Dict:
    """
    Calculate metrics for both unconstrained and constrained predictions.
    
    Returns
    -------
    Dict with structure:
    {
        'unconstrained': {metrics},
        'constrained': {metrics},
        'improvement': {delta metrics}
    }
    """
    
    def calc_metrics(pred, true, var_name):
        pred_flat = pred.flatten()
        true_flat = true.flatten()
        
        valid = ~(np.isnan(pred_flat) | np.isnan(true_flat) | 
                 np.isinf(pred_flat) | np.isinf(true_flat))
        
        pred_valid = pred_flat[valid]
        true_valid = true_flat[valid]
        
        if len(pred_valid) == 0:
            return {'r2': np.nan, 'rmse': np.nan, 'mae': np.nan, 'n': 0}
        
        r2 = r2_score(true_valid, pred_valid)
        rmse = np.sqrt(mean_squared_error(true_valid, pred_valid))
        mae = np.mean(np.abs(pred_valid - true_valid))
        bias = np.mean(pred_valid - true_valid)
        
        # Count negative predictions
        n_negative = np.sum(pred_valid < 0)
        pct_negative = 100 * n_negative / len(pred_valid)
        
        # Variance ratio
        var_pred = np.var(pred_valid)
        var_true = np.var(true_valid)
        var_ratio = var_pred / var_true if var_true > 0 else np.nan
        
        return {
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'bias': float(bias),
            'n_valid': int(len(pred_valid)),
            'n_negative': int(n_negative),
            'pct_negative': float(pct_negative),
            'variance_ratio': float(var_ratio)
        }
    
    comparison = {
        'unconstrained': {},
        'constrained': {},
        'improvement': {}
    }
    
    for var in ['visc_coeff', 'diff_coeff']:
        comparison['unconstrained'][var] = calc_metrics(
            predictions_unconstrained[var], truth[var], var
        )
        comparison['constrained'][var] = calc_metrics(
            predictions_constrained[var], truth[var], var
        )
        
        # Calculate improvement
        r2_unc = comparison['unconstrained'][var]['r2']
        r2_con = comparison['constrained'][var]['r2']
        rmse_unc = comparison['unconstrained'][var]['rmse']
        rmse_con = comparison['constrained'][var]['rmse']
        
        comparison['improvement'][var] = {
            'delta_r2': float(r2_con - r2_unc),
            'delta_rmse': float(rmse_con - rmse_unc),
            'pct_r2_improvement': float(100 * (r2_con - r2_unc) / abs(r2_unc)) if r2_unc != 0 else np.nan,
            'pct_rmse_improvement': float(100 * (rmse_unc - rmse_con) / rmse_unc) if rmse_unc != 0 else np.nan
        }
    
    return comparison


# ===================================================================
#  NON-ZERO METRICS WITH CONSTRAINTS
# ===================================================================

def calculate_nonzero_metrics_with_constraints(predictions_unconstrained: Dict,
                                               predictions_constrained: Dict,
                                               truth: Dict) -> Dict:
    """
    Calculate non-zero metrics for both unconstrained and constrained predictions.
    """
    
    nz_unconstrained = calculate_nonzero_metrics(predictions_unconstrained, truth)
    nz_constrained = calculate_nonzero_metrics(predictions_constrained, truth)
    
    comparison = {
        'unconstrained': nz_unconstrained,
        'constrained': nz_constrained,
        'improvement': {}
    }
    
    for var in ['visc_coeff', 'diff_coeff']:
        if var not in nz_unconstrained or var not in nz_constrained:
            continue
        
        # All data improvement
        r2_all_unc = nz_unconstrained[var]['all']['r2']
        r2_all_con = nz_constrained[var]['all']['r2']
        rmse_all_unc = nz_unconstrained[var]['all']['rmse']
        rmse_all_con = nz_constrained[var]['all']['rmse']
        
        # Non-zero only improvement
        r2_nz_unc = nz_unconstrained[var]['nonzero']['r2']
        r2_nz_con = nz_constrained[var]['nonzero']['r2']
        rmse_nz_unc = nz_unconstrained[var]['nonzero']['rmse']
        rmse_nz_con = nz_constrained[var]['nonzero']['rmse']
        
        comparison['improvement'][var] = {
            'all_data': {
                'delta_r2': float(r2_all_con - r2_all_unc),
                'delta_rmse': float(rmse_all_con - rmse_all_unc),
                'pct_r2_change': float(100 * (r2_all_con - r2_all_unc) / abs(r2_all_unc)) if r2_all_unc != 0 else np.nan
            },
            'nonzero_only': {
                'delta_r2': float(r2_nz_con - r2_nz_unc),
                'delta_rmse': float(rmse_nz_con - rmse_nz_unc),
                'pct_r2_change': float(100 * (r2_nz_con - r2_nz_unc) / abs(r2_nz_unc)) if r2_nz_unc != 0 else np.nan
            }
        }
    
    return comparison


# ===================================================================
#  VISUALIZATION FUNCTIONS
# ===================================================================

def create_comprehensive_comparison_table(all_comparisons: Dict, output_dir: Path):
    """Create comparison table for baseline and/or Ri-conditioned models."""
    
    output_dir = Path(output_dir)
    rows = []
    
    for model_name in sorted(all_comparisons.keys()):
        comp = all_comparisons[model_name]
        
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
        
        for var in ['visc_coeff', 'diff_coeff']:
            unc = comp['unconstrained'][var]
            con = comp['constrained'][var]
            imp = comp['improvement'][var]
            
            var_label = var.replace('_coeff', '').upper()
            
            rows.append({
                'Model_Type': model_type,
                'Architecture': arch,
                'Variable': var_label,
                'Constraint': 'Unconstrained',
                'R²': unc['r2'],
                'RMSE': unc['rmse'],
                'MAE': unc['mae'],
                'Bias': unc['bias'],
                'Var_Ratio': unc['variance_ratio'],
                '% Negative': unc['pct_negative']
            })
            
            rows.append({
                'Model_Type': model_type,
                'Architecture': arch,
                'Variable': var_label,
                'Constraint': 'Constrained (≥0)',
                'R²': con['r2'],
                'RMSE': con['rmse'],
                'MAE': con['mae'],
                'Bias': con['bias'],
                'Var_Ratio': con['variance_ratio'],
                '% Negative': con['pct_negative']
            })
    
    df = pd.DataFrame(rows)
    
    # Save as CSV
    csv_file = output_dir / 'constraint_comparison_fixed.csv'
    df.to_csv(csv_file, index=False, float_format='%.4f')
    logger.info(f"✓ Saved comparison table: {csv_file}")
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("CONSTRAINT IMPACT SUMMARY (FIXED)")
    logger.info(f"{'='*80}\n")
    
    for model_name in sorted(all_comparisons.keys()):
        logger.info(f"{model_name}:")
        comp = all_comparisons[model_name]
        
        for var in ['visc_coeff', 'diff_coeff']:
            var_label = var.replace('_coeff', '').upper()
            unc = comp['unconstrained'][var]
            con = comp['constrained'][var]
            imp = comp['improvement'][var]
            
            logger.info(f"\n  {var_label}:")
            logger.info(f"    Unconstrained:  R²={unc['r2']:7.4f}, RMSE={unc['rmse']:7.4f}, "
                       f"Bias={unc['bias']:+7.4f}, Var_Ratio={unc['variance_ratio']:.3f}, "
                       f"{unc['pct_negative']:5.2f}% negative")
            logger.info(f"    Constrained:    R²={con['r2']:7.4f}, RMSE={con['rmse']:7.4f}, "
                       f"Bias={con['bias']:+7.4f}, Var_Ratio={con['variance_ratio']:.3f}, "
                       f"{con['pct_negative']:5.2f}% negative")
            
            # Improvement indicators
            r2_arrow = '↑' if imp['delta_r2'] > 0 else '↓'
            rmse_arrow = '↓' if imp['delta_rmse'] < 0 else '↑'
            
            logger.info(f"    Δ R²:  {r2_arrow} {imp['delta_r2']:+.4f} ({imp['pct_r2_improvement']:+.1f}%)")
            logger.info(f"    Δ RMSE: {rmse_arrow} {imp['delta_rmse']:+.4f} ({imp['pct_rmse_improvement']:+.1f}%)")
    
    logger.info(f"\n{'='*80}\n")
    
    return df


def plot_constraint_comparison_scatter(predictions_unc: Dict,
                                       predictions_con: Dict,
                                       truth: Dict,
                                       output_dir: Path,
                                       model_name: str):
    """Create side-by-side scatter plots: unconstrained vs constrained."""
    
    output_dir = Path(output_dir)
    
    for var_key in ['visc_coeff', 'diff_coeff']:
        var_label = var_key.replace('_coeff', ' Coefficient').title()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Unconstrained
        ax = axes[0]
        pred_flat = predictions_unc[var_key].flatten()
        true_flat = truth[var_key].flatten()
        
        valid = ~(np.isnan(pred_flat) | np.isnan(true_flat) | 
                 np.isinf(pred_flat) | np.isinf(true_flat))
        pred_valid = pred_flat[valid]
        true_valid = true_flat[valid]
        
        if len(pred_valid) > 0:
            data_min = np.percentile(true_valid, 1)
            data_max = np.percentile(true_valid, 99)
            
            hexbin = ax.hexbin(true_valid, pred_valid, gridsize=60, cmap='Reds',
                             norm=LogNorm(vmin=1, vmax=max(10, len(pred_valid)/100)),
                             mincnt=1, alpha=0.9)
            
            cbar = plt.colorbar(hexbin, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Count (log)', fontsize=9)
            
            ax.plot([data_min, data_max], [data_min, data_max], 
                   'k--', linewidth=2, alpha=0.6, label='1:1', zorder=10)
            
            # Show negative region
            ax.axhline(y=0, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='Zero line')
            ax.axvline(x=0, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
            
            r2 = r2_score(true_valid, pred_valid)
            rmse = np.sqrt(mean_squared_error(true_valid, pred_valid))
            bias = np.mean(pred_valid - true_valid)
            n_neg = np.sum(pred_valid < 0)
            pct_neg = 100 * n_neg / len(pred_valid)
            
            stats = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nBias = {bias:+.4f}\n{n_neg:,} negative ({pct_neg:.2f}%)'
            ax.text(0.03, 0.97, stats, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                   family='monospace')
            
            ax.set_title('Unconstrained', fontsize=12, fontweight='bold', color='#E63946')
        
        # Constrained
        ax = axes[1]
        pred_flat = predictions_con[var_key].flatten()
        true_flat = truth[var_key].flatten()
        
        valid = ~(np.isnan(pred_flat) | np.isnan(true_flat) | 
                 np.isinf(pred_flat) | np.isinf(true_flat))
        pred_valid = pred_flat[valid]
        true_valid = true_flat[valid]
        
        if len(pred_valid) > 0:
            data_min = np.percentile(true_valid, 1)
            data_max = np.percentile(true_valid, 99)
            
            hexbin = ax.hexbin(true_valid, pred_valid, gridsize=60, cmap='Greens',
                             norm=LogNorm(vmin=1, vmax=max(10, len(pred_valid)/100)),
                             mincnt=1, alpha=0.9)
            
            cbar = plt.colorbar(hexbin, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Count (log)', fontsize=9)
            
            ax.plot([data_min, data_max], [data_min, data_max], 
                   'k--', linewidth=2, alpha=0.6, label='1:1', zorder=10)
            
            ax.axhline(y=0, color='green', linestyle=':', linewidth=1.5, alpha=0.5, label='Zero line')
            ax.axvline(x=0, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
            
            r2 = r2_score(true_valid, pred_valid)
            rmse = np.sqrt(mean_squared_error(true_valid, pred_valid))
            bias = np.mean(pred_valid - true_valid)
            n_neg = np.sum(pred_valid < 0)
            pct_neg = 100 * n_neg / len(pred_valid)
            
            stats = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nBias = {bias:+.4f}\n{n_neg:,} negative ({pct_neg:.2f}%)'
            ax.text(0.03, 0.97, stats, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                   family='monospace')
            
            ax.set_title('Constrained (≥0)', fontsize=12, fontweight='bold', color='#06A77D')
        
        # Common formatting
        for ax in axes:
            ax.set_xlabel('MONC Truth', fontsize=10, fontweight='bold')
            ax.set_ylabel('Model Prediction', fontsize=10, fontweight='bold')
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
        
        fig.suptitle(f'{model_name} - {var_label}: Constraint Impact (FIXED)',
                    fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        output_file = output_dir / f'constraint_comparison_fixed_{model_name}_{var_key}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"  ✓ Saved: {output_file.name}")


def plot_constraint_improvement_heatmap(all_comparisons: Dict, output_dir: Path):
    """Create heatmap showing R² improvement from constraints."""
    
    output_dir = Path(output_dir)
    
    model_names = sorted(all_comparisons.keys())
    variables = ['visc_coeff', 'diff_coeff']
    var_labels = ['Viscosity', 'Diffusivity']
    
    improvements = np.zeros((len(model_names), len(variables)))
    
    for i, model_name in enumerate(model_names):
        for j, var in enumerate(variables):
            improvements[i, j] = all_comparisons[model_name]['improvement'][var]['delta_r2']
    
    fig, ax = plt.subplots(figsize=(8, max(6, len(model_names) * 0.8)))
    
    im = ax.imshow(improvements, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.1)
    
    ax.set_xticks(range(len(variables)))
    ax.set_xticklabels(var_labels, fontsize=11)
    ax.set_yticks(range(len(model_names)))
    
    display_names = [name.replace('Baseline-', 'B-').replace('Ri-', 'Ri-') for name in model_names]
    ax.set_yticklabels(display_names, fontsize=11)
    
    for i in range(len(model_names)):
        for j in range(len(variables)):
            val = improvements[i, j]
            color = 'white' if abs(val) > 0.05 else 'black'
            ax.text(j, i, f'{val:+.4f}', ha='center', va='center', 
                   color=color, fontweight='bold', fontsize=10)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('ΔR² (Constrained - Unconstrained)', fontsize=10, fontweight='bold')
    
    ax.set_title('Impact of Non-Negativity Constraint on R² (FIXED)', fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Coefficient', fontsize=11, fontweight='bold')
    ax.set_ylabel('Model', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = output_dir / 'constraint_improvement_heatmap_fixed.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved improvement heatmap: {output_file.name}")


def plot_comparison_across_model_types(all_comparisons: Dict, output_dir: Path):
    """Compare constraint impact between baseline and Ri-conditioned models."""
    
    output_dir = Path(output_dir)
    
    baseline_models = {k: v for k, v in all_comparisons.items() if k.startswith('Baseline-')}
    ri_models = {k: v for k, v in all_comparisons.items() if k.startswith('Ri-')}
    
    if not (baseline_models and ri_models):
        logger.info("Skipping model type comparison (need both baseline and Ri-conditioned)")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, var in enumerate(['visc_coeff', 'diff_coeff']):
        ax = axes[idx]
        
        architectures = ['MLP', 'ResMLP', 'TabTransformer']
        x_pos = np.arange(len(architectures))
        width = 0.35
        
        baseline_deltas = []
        ri_deltas = []
        
        for arch in architectures:
            baseline_key = f'Baseline-{arch}'
            ri_key = f'Ri-{arch}'
            
            if baseline_key in baseline_models:
                baseline_deltas.append(baseline_models[baseline_key]['improvement'][var]['delta_r2'])
            else:
                baseline_deltas.append(0)
            
            if ri_key in ri_models:
                ri_deltas.append(ri_models[ri_key]['improvement'][var]['delta_r2'])
            else:
                ri_deltas.append(0)
        
        bars1 = ax.bar(x_pos - width/2, baseline_deltas, width, 
                      label='Baseline', alpha=0.8, color='#2E86AB')
        bars2 = ax.bar(x_pos + width/2, ri_deltas, width,
                      label='Ri-Conditioned', alpha=0.8, color='#E63946')
        
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:+.4f}', ha='center', va='bottom' if height > 0 else 'top', 
                   fontsize=8)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Architecture', fontsize=11, fontweight='bold')
        ax.set_ylabel('ΔR² (Constrained - Unconstrained)', fontsize=11, fontweight='bold')
        ax.set_title(f'{var.replace("_coeff", "").upper()}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(architectures, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_file = output_dir / 'constraint_impact_baseline_vs_ri_fixed.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved model type comparison: {output_file.name}")


# ===================================================================
#  NON-ZERO COMPARISON FUNCTIONS (from previous implementation)
# ===================================================================

def create_constraint_nonzero_comparison_table(all_comparisons: Dict, output_dir: Path):
    """Create comprehensive table showing constraint impact on all-data vs non-zero."""
    
    output_dir = Path(output_dir)
    rows = []
    
    for model_name in sorted(all_comparisons.keys()):
        comp = all_comparisons[model_name]
        
        if model_name.startswith('Baseline-'):
            model_type = 'Baseline'
            arch = model_name.replace('Baseline-', '')
        elif model_name.startswith('Ri-'):
            model_type = 'Ri-Conditioned'
            arch = model_name.replace('Ri-', '')
        else:
            model_type = 'Unknown'
            arch = model_name
        
        for var in ['visc_coeff', 'diff_coeff']:
            if var not in comp['unconstrained']:
                continue
            
            var_label = var.replace('_coeff', '').upper()
            
            # Unconstrained - All Data
            unc_all = comp['unconstrained'][var]['all']
            rows.append({
                'Model_Type': model_type,
                'Architecture': arch,
                'Variable': var_label,
                'Constraint': 'Unconstrained',
                'Subset': 'All Data',
                'R²': unc_all['r2'],
                'RMSE': unc_all['rmse'],
                'Bias': unc_all['bias'],
                'N_Points': unc_all['n_valid']
            })
            
            # Unconstrained - Non-Zero
            unc_nz = comp['unconstrained'][var]['nonzero']
            rows.append({
                'Model_Type': model_type,
                'Architecture': arch,
                'Variable': var_label,
                'Constraint': 'Unconstrained',
                'Subset': 'Non-Zero Only',
                'R²': unc_nz['r2'],
                'RMSE': unc_nz['rmse'],
                'Bias': unc_nz['bias'],
                'N_Points': unc_nz['n_valid']
            })
            
            # Constrained - All Data
            con_all = comp['constrained'][var]['all']
            rows.append({
                'Model_Type': model_type,
                'Architecture': arch,
                'Variable': var_label,
                'Constraint': 'Constrained (≥0)',
                'Subset': 'All Data',
                'R²': con_all['r2'],
                'RMSE': con_all['rmse'],
                'Bias': con_all['bias'],
                'N_Points': con_all['n_valid']
            })
            
            # Constrained - Non-Zero
            con_nz = comp['constrained'][var]['nonzero']
            rows.append({
                'Model_Type': model_type,
                'Architecture': arch,
                'Variable': var_label,
                'Constraint': 'Constrained (≥0)',
                'Subset': 'Non-Zero Only',
                'R²': con_nz['r2'],
                'RMSE': con_nz['rmse'],
                'Bias': con_nz['bias'],
                'N_Points': con_nz['n_valid']
            })
    
    df = pd.DataFrame(rows)
    
    csv_file = output_dir / 'constraint_nonzero_comparison_fixed.csv'
    df.to_csv(csv_file, index=False, float_format='%.4f')
    logger.info(f"✓ Saved: {csv_file.name}")
    
    # Print summary
    logger.info(f"\n{'='*90}")
    logger.info("CONSTRAINT IMPACT ON NON-ZERO DATA (FIXED)")
    logger.info(f"{'='*90}\n")
    
    for model_name in sorted(all_comparisons.keys()):
        logger.info(f"{model_name}:")
        comp = all_comparisons[model_name]
        
        for var in ['visc_coeff', 'diff_coeff']:
            if var not in comp['improvement']:
                continue
            
            var_label = var.replace('_coeff', '').upper()
            imp = comp['improvement'][var]
            zero_frac = comp['unconstrained'][var]['zero_fraction']
            
            logger.info(f"\n  {var_label} (Zero fraction: {zero_frac*100:.2f}%):")
            logger.info(f"    ALL DATA:")
            logger.info(f"      ΔR²: {imp['all_data']['delta_r2']:+.4f}")
            logger.info(f"      ΔRMSE: {imp['all_data']['delta_rmse']:+.4f}")
            
            logger.info(f"    NON-ZERO ONLY:")
            logger.info(f"      ΔR²: {imp['nonzero_only']['delta_r2']:+.4f}")
            logger.info(f"      ΔRMSE: {imp['nonzero_only']['delta_rmse']:+.4f}")
    
    logger.info(f"\n{'='*90}\n")
    
    return df


def plot_constraint_impact_by_subset(all_comparisons: Dict, output_dir: Path):
    """Visualize how constraints affect all-data vs non-zero performance."""
    
    output_dir = Path(output_dir)
    
    logger.info("\nCreating constraint impact by subset plots...")
    
    model_names = sorted(all_comparisons.keys())
    
    for var in ['visc_coeff', 'diff_coeff']:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        var_label = var.replace('_coeff', '').title()
        
        # R² improvement comparison
        ax = axes[0]
        x_pos = np.arange(len(model_names))
        width = 0.35
        
        delta_r2_all = []
        delta_r2_nz = []
        
        for model_name in model_names:
            if var in all_comparisons[model_name]['improvement']:
                delta_r2_all.append(all_comparisons[model_name]['improvement'][var]['all_data']['delta_r2'])
                delta_r2_nz.append(all_comparisons[model_name]['improvement'][var]['nonzero_only']['delta_r2'])
            else:
                delta_r2_all.append(0)
                delta_r2_nz.append(0)
        
        bars1 = ax.bar(x_pos - width/2, delta_r2_all, width, 
                      label='All Data', alpha=0.8, color='#2E86AB')
        bars2 = ax.bar(x_pos + width/2, delta_r2_nz, width,
                      label='Non-Zero Only', alpha=0.8, color='#E63946')
        
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:+.4f}', ha='center', va='bottom' if height > 0 else 'top', 
                   fontsize=8)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel('ΔR² (Constrained - Unconstrained)', fontsize=11, fontweight='bold')
        ax.set_title(f'{var_label} - Constraint Impact on R²', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('Baseline-', 'B-').replace('Ri-', 'Ri-') for m in model_names], 
                          fontsize=9, rotation=15, ha='right')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # RMSE improvement comparison
        ax = axes[1]
        
        delta_rmse_all = []
        delta_rmse_nz = []
        
        for model_name in model_names:
            if var in all_comparisons[model_name]['improvement']:
                delta_rmse_all.append(all_comparisons[model_name]['improvement'][var]['all_data']['delta_rmse'])
                delta_rmse_nz.append(all_comparisons[model_name]['improvement'][var]['nonzero_only']['delta_rmse'])
            else:
                delta_rmse_all.append(0)
                delta_rmse_nz.append(0)
        
        bars1 = ax.bar(x_pos - width/2, delta_rmse_all, width, 
                      label='All Data', alpha=0.8, color='#2E86AB')
        bars2 = ax.bar(x_pos + width/2, delta_rmse_nz, width,
                      label='Non-Zero Only', alpha=0.8, color='#E63946')
        
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:+.2f}', ha='center', va='bottom' if height > 0 else 'top', 
                   fontsize=8)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel('ΔRMSE (Constrained - Unconstrained)', fontsize=11, fontweight='bold')
        ax.set_title(f'{var_label} - Constraint Impact on RMSE', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('Baseline-', 'B-').replace('Ri-', 'Ri-') for m in model_names], 
                          fontsize=9, rotation=15, ha='right')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_file = output_dir / f'constraint_impact_by_subset_fixed_{var}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"  ✓ Saved: {output_file.name}")


# ===================================================================
#  MAIN FUNCTION
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Constrained Inference Analysis with FIXED constraint application'
    )
    
    parser.add_argument('--mode', type=str, required=True, choices=['single', 'timeseries'])
    parser.add_argument('--nc-file', type=Path, help='NetCDF file (for single mode)')
    parser.add_argument('--data-dir', type=Path, help='Directory with NetCDF files (for timeseries mode)')
    parser.add_argument('--output', type=Path, required=True, help='Output directory')
    
    # BASELINE model paths
    parser.add_argument('--baseline-mlp', type=Path, default=None)
    parser.add_argument('--baseline-resmlp', type=Path, default=None)
    parser.add_argument('--baseline-tabtransformer', type=Path, default=None)
    
    # RI-CONDITIONED model paths
    parser.add_argument('--ri-mlp', type=Path, default=None)
    parser.add_argument('--ri-resmlp', type=Path, default=None)
    parser.add_argument('--ri-tabtransformer', type=Path, default=None)
    
    parser.add_argument('--scaler-dir', type=Path, required=True, help='Directory with scalers')
    parser.add_argument('--time-idx', type=int, default=0, help='Time index in NetCDF')
    parser.add_argument('--k-min', type=int, default=0, help='Minimum vertical level')
    parser.add_argument('--k-max', type=int, default=219, help='Maximum vertical level')
    parser.add_argument('--n-workers', type=int, default=None, help='Number of worker processes')
    
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
        parser.error("Must provide at least one model (baseline or Ri-conditioned)")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize engine
    logger.info(f"\n{'='*80}")
    logger.info("CONSTRAINED INFERENCE ANALYSIS (FIXED VERSION)")
    logger.info(f"{'='*80}\n")
    logger.info("⚠️  CRITICAL FIX: Constraints applied AFTER inverse transform")
    logger.info("    Previous version applied constraints to Z-scores (incorrect)")
    logger.info("    This version applies constraints in physical space (correct)\n")
    
    engine = UnifiedInferenceEngine(
        baseline_paths=baseline_paths if baseline_paths else None,
        ri_paths=ri_paths if ri_paths else None,
        scaler_dir=args.scaler_dir,
        n_workers=args.n_workers
    )
    
    # Single file mode
    if args.mode == 'single':
        logger.info(f"Processing: {args.nc_file}\n")
        
        # Run inference (predictions are ALREADY in physical space from engine)
        predictions = engine.predict_3d_domain(args.nc_file, args.time_idx, args.k_min, args.k_max)
        truth = extract_truth_from_netcdf(args.nc_file, args.time_idx, args.k_min, args.k_max)
        
        # Compare constrained vs unconstrained for each model
        all_comparisons = {}
        all_nonzero_comparisons = {}
        
        for model_name in engine.models.keys():
            logger.info(f"\n{model_name}:")
            logger.info("-" * 60)
            
            pred_unc = predictions[model_name]
            
            # Apply constraints IN PHYSICAL SPACE
            pred_con = apply_physical_constraints(pred_unc, verbose=True)
            
            # Point-wise metrics
            comparison = calculate_metrics_comparison(pred_unc, pred_con, truth)
            all_comparisons[model_name] = comparison
            
            # Non-zero metrics
            nonzero_comparison = calculate_nonzero_metrics_with_constraints(pred_unc, pred_con, truth)
            all_nonzero_comparisons[model_name] = nonzero_comparison
            
            # Create comparison plots
            plot_constraint_comparison_scatter(pred_unc, pred_con, truth, output_dir, model_name)
        
        # Summary tables
        logger.info("\n" + "="*80)
        logger.info("GENERATING SUMMARY TABLES AND PLOTS")
        logger.info("="*80)
        
        df = create_comprehensive_comparison_table(all_comparisons, output_dir)
        plot_constraint_improvement_heatmap(all_comparisons, output_dir)
        
        if baseline_paths and ri_paths:
            plot_comparison_across_model_types(all_comparisons, output_dir)
        
        # Non-zero analysis
        logger.info("\n" + "="*80)
        logger.info("NON-ZERO CONSTRAINT ANALYSIS")
        logger.info("="*80)
        
        df_nonzero = create_constraint_nonzero_comparison_table(all_nonzero_comparisons, output_dir)
        plot_constraint_impact_by_subset(all_nonzero_comparisons, output_dir)
        
        # Standard non-zero plots
        all_unc_nz = {k: v['unconstrained'] for k, v in all_nonzero_comparisons.items()}
        all_con_nz = {k: v['constrained'] for k, v in all_nonzero_comparisons.items()}
        
        unc_dir = output_dir / 'unconstrained'
        con_dir = output_dir / 'constrained'
        unc_dir.mkdir(parents=True, exist_ok=True)
        con_dir.mkdir(parents=True, exist_ok=True)
        
        plot_nonzero_comparison(all_unc_nz, unc_dir)
        plot_nonzero_comparison(all_con_nz, con_dir)
        plot_zero_fraction_analysis(all_unc_nz, output_dir)
        
        # Save results
        with open(output_dir / 'constraint_analysis_fixed.json', 'w') as f:
            json.dump(all_comparisons, f, indent=2, cls=NumpyEncoder)
        
        with open(output_dir / 'constraint_nonzero_analysis_fixed.json', 'w') as f:
            json.dump(all_nonzero_comparisons, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"\n✅ Single file analysis complete: {output_dir}\n")
    
    # Timeseries mode
    elif args.mode == 'timeseries':
        nc_files = sorted(
            args.data_dir.glob('*.nc'),
            key=lambda p: int(re.search(r'(\d+)\.nc$', p.name).group(1))
        )
        
        logger.info(f"Processing {len(nc_files)} files...\n")
        
        # Aggregate storage
        model_names = list(engine.models.keys())
        agg_pred_unc = {m: {'visc_coeff': [], 'diff_coeff': []} for m in model_names}
        agg_pred_con = {m: {'visc_coeff': [], 'diff_coeff': []} for m in model_names}
        agg_truth = {'visc_coeff': [], 'diff_coeff': []}
        
        for nc_file in nc_files:
            logger.info(f"  {nc_file.name}")
            
            predictions = engine.predict_3d_domain(nc_file, args.time_idx, args.k_min, args.k_max)
            truth = extract_truth_from_netcdf(nc_file, args.time_idx, args.k_min, args.k_max)
            
            # Aggregate
            for var in ['visc_coeff', 'diff_coeff']:
                agg_truth[var].append(truth[var].flatten())
                
                for model_name in model_names:
                    agg_pred_unc[model_name][var].append(predictions[model_name][var].flatten())
                    
                    pred_con = apply_physical_constraints(predictions[model_name], verbose=False)
                    agg_pred_con[model_name][var].append(pred_con[var].flatten())
        
        # Concatenate
        logger.info("\nAggregating results...")
        for var in ['visc_coeff', 'diff_coeff']:
            agg_truth[var] = np.concatenate(agg_truth[var])
            for model_name in model_names:
                agg_pred_unc[model_name][var] = np.concatenate(agg_pred_unc[model_name][var])
                agg_pred_con[model_name][var] = np.concatenate(agg_pred_con[model_name][var])
        
        # Calculate metrics
        all_comparisons = {}
        all_nonzero_comparisons = {}
        
        for model_name in model_names:
            comparison = calculate_metrics_comparison(
                agg_pred_unc[model_name],
                agg_pred_con[model_name],
                agg_truth
            )
            all_comparisons[model_name] = comparison
            
            nonzero_comparison = calculate_nonzero_metrics_with_constraints(
                agg_pred_unc[model_name],
                agg_pred_con[model_name],
                agg_truth
            )
            all_nonzero_comparisons[model_name] = nonzero_comparison
            
            # Plots
            predictions_3d = engine.predict_3d_domain(nc_files[0], args.time_idx, args.k_min, args.k_max)
            truth_3d = extract_truth_from_netcdf(nc_files[0], args.time_idx, args.k_min, args.k_max)
            
            pred_unc_3d = predictions_3d[model_name]
            pred_con_3d = apply_physical_constraints(pred_unc_3d, verbose=False)
            
            plot_constraint_comparison_scatter(pred_unc_3d, pred_con_3d, truth_3d, output_dir, model_name)
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("GENERATING SUMMARY TABLES AND PLOTS")
        logger.info("="*80)
        
        df = create_comprehensive_comparison_table(all_comparisons, output_dir)
        plot_constraint_improvement_heatmap(all_comparisons, output_dir)
        
        if baseline_paths and ri_paths:
            plot_comparison_across_model_types(all_comparisons, output_dir)
        
        # Non-zero
        logger.info("\n" + "="*80)
        logger.info("NON-ZERO CONSTRAINT ANALYSIS (AGGREGATE)")
        logger.info("="*80)
        
        df_nonzero = create_constraint_nonzero_comparison_table(all_nonzero_comparisons, output_dir)
        plot_constraint_impact_by_subset(all_nonzero_comparisons, output_dir)
        
        all_unc_nz = {k: v['unconstrained'] for k, v in all_nonzero_comparisons.items()}
        all_con_nz = {k: v['constrained'] for k, v in all_nonzero_comparisons.items()}
        
        unc_dir = output_dir / 'unconstrained'
        con_dir = output_dir / 'constrained'
        unc_dir.mkdir(parents=True, exist_ok=True)
        con_dir.mkdir(parents=True, exist_ok=True)
        
        plot_nonzero_comparison(all_unc_nz, unc_dir)
        plot_nonzero_comparison(all_con_nz, con_dir)
        plot_zero_fraction_analysis(all_unc_nz, output_dir)
        
        # Save results
        with open(output_dir / 'constraint_analysis_aggregate_fixed.json', 'w') as f:
            json.dump(all_comparisons, f, indent=2, cls=NumpyEncoder)
        
        with open(output_dir / 'constraint_nonzero_analysis_aggregate_fixed.json', 'w') as f:
            json.dump(all_nonzero_comparisons, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"\n✅ Timeseries analysis complete: {output_dir}\n")


if __name__ == '__main__':
    main()
