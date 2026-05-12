#!/usr/bin/env python3
"""
Comparative KDE Analysis: Log-Space Optimized
=============================================

Generates KDE plots in LOG-10 SPACE to handle heavy-tailed coefficient distributions.
Visualizes Log10(Prediction) vs Log10(Truth).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import r2_score
import re
import warnings

# Import your existing engine
from run_models_comparison_with_ri_v2 import UnifiedInferenceEngine
from run_best_models_analysis import extract_truth_from_netcdf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="paper")
warnings.filterwarnings("ignore")

def plot_comparative_kde_log(predictions, truth, output_dir, file_count, plot_points=5_000_000):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    architectures = ['MLP', 'ResMLP', 'TabTransformer']
    types = ['Baseline', 'Ri']
    
    model_map = {
        'Baseline': {'MLP': 'Baseline-MLP', 'ResMLP': 'Baseline-ResMLP', 'TabTransformer': 'Baseline-TabTransformer'},
        'Ri': {'MLP': 'Ri-MLP', 'ResMLP': 'Ri-ResMLP', 'TabTransformer': 'Ri-TabTransformer'}
    }
    colors = {'Baseline': '#2E86AB', 'Ri': '#E63946'}

    for var in ['visc_coeff', 'diff_coeff']:
        var_label = var.replace('_coeff', '').title()
        logger.info(f"Generating Log-Space KDE for {var_label}...")

        # --- LOG TRANSFORM TRUTH ---
        true_raw = truth[var]
        # Filter: strictly positive values for Log10 (avoid -inf)
        epsilon = 1e-6 
        true_log = np.log10(np.maximum(true_raw, epsilon))

        # Subsample for Plotting
        if len(true_log) > plot_points:
            indices = np.random.choice(len(true_log), plot_points, replace=False)
            true_plot = true_log[indices]
        else:
            indices = slice(None)
            true_plot = true_log
            
        # Determine shared limits in Log Space
        xmin, xmax = np.nanpercentile(true_log, [0.1, 99.9])
        
        # Grid Setup
        fig, axes = plt.subplots(3, 2, figsize=(14, 16), sharex=True, sharey=True)
        
        for i, arch in enumerate(architectures):
            for j, model_type in enumerate(types):
                ax = axes[i, j]
                model_key = model_map[model_type][arch]
                
                if model_key not in predictions:
                    ax.text(0.5, 0.5, "Model Not Loaded", ha='center', va='center'); continue
                
                # --- LOG TRANSFORM PREDICTION ---
                pred_raw = predictions[model_key][var]
                pred_log = np.log10(np.maximum(pred_raw, epsilon))
                
                # Metrics (Calculated on LOG data for shape match, or Raw for R2)
                # Let's show R2 on RAW data, but KS/W on Log data (shape)
                valid = ~(np.isnan(true_raw) | np.isnan(pred_raw))
                r2 = r2_score(true_raw[valid], pred_raw[valid])
                
                valid_log = ~(np.isnan(true_log) | np.isnan(pred_log))
                ks_stat, _ = ks_2samp(true_log[valid_log], pred_log[valid_log])
                wd_stat = wasserstein_distance(true_log[valid_log], pred_log[valid_log])
                
                # --- PLOTTING ---
                pred_plot = pred_log[indices]

                # Truth (Black Dashed)
                sns.kdeplot(true_plot, ax=ax, color='black', linestyle='--', linewidth=2, 
                           label='MONC Truth', clip=(xmin, xmax))
                
                # Prediction (Colored)
                color = colors[model_type]
                sns.kdeplot(pred_plot, ax=ax, color=color, linewidth=2.5, fill=True, alpha=0.1,
                           label='Prediction', clip=(xmin, xmax))
                
                ax.set_title(f"{model_type} - {arch}", fontsize=12, fontweight='bold')
                ax.set_xlim(xmin, xmax)
                
                # Stats Box
                stats = (f"Raw R² = {r2:.3f}\n"
                         f"Log KS = {ks_stat:.3f}\n"
                         f"Log WD = {wd_stat:.3f}")
                
                ax.text(0.05, 0.95, stats, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.grid(True, alpha=0.3)
                if i == 0 and j == 0: ax.legend(loc='upper right')
                if i == 2: ax.set_xlabel(f"Log10({var_label})")

        plt.suptitle(f"{var_label} (Log-Space): Global Aggregate ({file_count} files)", 
                    fontsize=16, fontweight='bold', y=0.92)
        
        outfile = output_dir / f"comparative_kde_log_{var}.png"
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        logger.info(f"Saved {outfile}")
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--scaler-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default='kde_log_results')
    parser.add_argument('--sample-rate', type=float, default=1.0)
    
    # Model Args
    parser.add_argument('--baseline-mlp', type=Path)
    parser.add_argument('--ri-mlp', type=Path)
    parser.add_argument('--baseline-resmlp', type=Path)
    parser.add_argument('--ri-resmlp', type=Path)
    parser.add_argument('--baseline-tabtransformer', type=Path)
    parser.add_argument('--ri-tabtransformer', type=Path)
    
    parser.add_argument('--k-min', type=int, default=0)
    parser.add_argument('--k-max', type=int, default=219)
    parser.add_argument('--time-idx', type=int, default=0)

    args = parser.parse_args()

    # Setup Models & Engine (Same as before)
    baseline_paths = {}
    if args.baseline_mlp: baseline_paths['MLP'] = args.baseline_mlp
    if args.baseline_resmlp: baseline_paths['ResMLP'] = args.baseline_resmlp
    if args.baseline_tabtransformer: baseline_paths['TabTransformer'] = args.baseline_tabtransformer

    ri_paths = {}
    if args.ri_mlp: ri_paths['MLP'] = args.ri_mlp
    if args.ri_resmlp: ri_paths['ResMLP'] = args.ri_resmlp
    if args.ri_tabtransformer: ri_paths['TabTransformer'] = args.ri_tabtransformer

    engine = UnifiedInferenceEngine(baseline_paths=baseline_paths, ri_paths=ri_paths, scaler_dir=args.scaler_dir)
    
    # File Aggregation (Same as before)
    nc_files = sorted(list(args.data_dir.glob('*.nc')))
    nc_files.sort(key=lambda f: int(re.search(r'(\d+)', f.name).group()) if re.search(r'(\d+)', f.name) else f.name)
    
    if not nc_files: return

    logger.info(f"Found {len(nc_files)} files. Aggregating...")

    agg_preds = {k: {'visc_coeff': [], 'diff_coeff': []} for k in engine.models.keys()}
    agg_truth = {'visc_coeff': [], 'diff_coeff': []}

    for nc_file in tqdm(nc_files):
        try:
            preds_file = engine.predict_3d_domain(nc_file, args.time_idx, args.k_min, args.k_max)
            truth_file = extract_truth_from_netcdf(nc_file, args.time_idx, args.k_min, args.k_max)
            
            flat_truth = truth_file['visc_coeff'].flatten()
            n_points = len(flat_truth)
            if args.sample_rate < 1.0:
                indices = np.random.choice(n_points, int(n_points * args.sample_rate), replace=False)
            else:
                indices = slice(None)
            
            for var in ['visc_coeff', 'diff_coeff']:
                agg_truth[var].append(truth_file[var].flatten()[indices])
            for model_key in engine.models.keys():
                for var in ['visc_coeff', 'diff_coeff']:
                    agg_preds[model_key][var].append(preds_file[model_key][var].flatten()[indices])
        except Exception: pass

    final_truth = {var: np.concatenate(agg_truth[var]) for var in agg_truth}
    final_preds = {k: {var: np.concatenate(agg_preds[k][var]) for var in agg_preds[k]} for k in agg_preds}

    # Plot
    plot_comparative_kde_log(final_preds, final_truth, args.output, len(nc_files))

if __name__ == "__main__":
    main()
