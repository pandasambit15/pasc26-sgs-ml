#!/usr/bin/env python3
"""
Active Metrics Calculator + Data Exporter
=========================================

1. Calculates validation metrics for "Active" turbulence (Truth > threshold).
2. OPTIONAL: Saves the processed FULL and FILTERED data arrays to .npz files.
   This allows strictly faster re-analysis later without reading NetCDFs.

Usage:
    python save_processed_metrics_and_data.py \
        --data-dir /path/to/inference_files/ \
        --scaler-dir scalers/ \
        --output results/ \
        --threshold 0.01 \
        --save-numpy  <-- ADDS DATA SAVING
        --baseline-mlp mlp.pth ...
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
from scipy.stats import ks_2samp, entropy
from sklearn.metrics import r2_score, mean_squared_error
import re
import warnings

# Import existing engine
from run_models_comparison_with_ri_v2 import UnifiedInferenceEngine
from run_best_models_analysis import extract_truth_from_netcdf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

def calculate_kld(p_samples, q_samples, bins=100):
    """Estimates KL Divergence D(P || Q)."""
    min_val = min(np.min(p_samples), np.min(q_samples))
    max_val = max(np.max(p_samples), np.max(q_samples))
    
    p_hist, _ = np.histogram(p_samples, bins=bins, range=(min_val, max_val), density=True)
    q_hist, _ = np.histogram(q_samples, bins=bins, range=(min_val, max_val), density=True)
    
    epsilon = 1e-10
    p_prob = p_hist + epsilon
    q_prob = q_hist + epsilon
    
    p_prob /= np.sum(p_prob)
    q_prob /= np.sum(q_prob)
    
    return entropy(p_prob, q_prob)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--scaler-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default='processed_data_and_metrics')
    parser.add_argument('--threshold', type=float, default=0.01, help="Threshold for active turbulence")
    parser.add_argument('--sample-rate', type=float, default=1.0)
    parser.add_argument('--save-numpy', action='store_true', help="Save extracted arrays to .npz")
    
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
    args.output.mkdir(parents=True, exist_ok=True)

    # 1. Setup Models
    baseline_paths = {}
    if args.baseline_mlp: baseline_paths['MLP'] = args.baseline_mlp
    if args.baseline_resmlp: baseline_paths['ResMLP'] = args.baseline_resmlp
    if args.baseline_tabtransformer: baseline_paths['TabTransformer'] = args.baseline_tabtransformer

    ri_paths = {}
    if args.ri_mlp: ri_paths['MLP'] = args.ri_mlp
    if args.ri_resmlp: ri_paths['ResMLP'] = args.ri_resmlp
    if args.ri_tabtransformer: ri_paths['TabTransformer'] = args.ri_tabtransformer

    engine = UnifiedInferenceEngine(baseline_paths=baseline_paths, ri_paths=ri_paths, scaler_dir=args.scaler_dir)
    
    # 2. Find Files
    nc_files = sorted(list(args.data_dir.glob('*.nc')))
    nc_files.sort(key=lambda f: int(re.search(r'(\d+)', f.name).group()) if re.search(r'(\d+)', f.name) else f.name)
    
    if not nc_files: return

    logger.info(f"Found {len(nc_files)} files. Processing...")

    # Data Containers
    agg_preds = {k: {'visc_coeff': [], 'diff_coeff': []} for k in engine.models.keys()}
    agg_truth = {'visc_coeff': [], 'diff_coeff': []}

    # 3. Load & Aggregate
    for nc_file in tqdm(nc_files):
        try:
            preds = engine.predict_3d_domain(nc_file, args.time_idx, args.k_min, args.k_max)
            truth = extract_truth_from_netcdf(nc_file, args.time_idx, args.k_min, args.k_max)
            
            n_points = truth['visc_coeff'].size
            if args.sample_rate < 1.0:
                idx = np.random.choice(n_points, int(n_points * args.sample_rate), replace=False)
            else:
                idx = slice(None)
                
            for var in ['visc_coeff', 'diff_coeff']:
                agg_truth[var].append(truth[var].flatten()[idx])
                for m in engine.models.keys():
                    agg_preds[m][var].append(preds[m][var].flatten()[idx])
        except Exception as e:
            logger.warning(f"Error {nc_file.name}: {e}")

    # 4. Process Variable by Variable
    metrics_results = []
    
    for var in ['visc_coeff', 'diff_coeff']:
        var_label = var.replace('_coeff', '').upper()
        logger.info(f"\nProcessing {var_label}...")
        
        # A. Create FULL DOMAIN Arrays
        full_truth = np.concatenate(agg_truth[var])
        full_data_dict = {'truth': full_truth}
        
        # B. Create ACTIVE MASK
        active_mask = full_truth > args.threshold
        truth_active = full_truth[active_mask]
        active_data_dict = {'truth': truth_active}
        
        active_count = len(truth_active)
        active_pct = 100 * active_count / len(full_truth)
        logger.info(f"  Active Data (> {args.threshold}): {active_count:,} points ({active_pct:.1f}%)")
        
        # Loop Models
        for model_name in engine.models.keys():
            # 1. Get Full Prediction
            full_pred = np.concatenate(agg_preds[model_name][var])
            full_data_dict[model_name] = full_pred
            
            # 2. Get Active Prediction (Using Truth Mask)
            pred_active = full_pred[active_mask]
            active_data_dict[model_name] = pred_active
            
            # 3. Calculate Metrics (On Active Data)
            r2 = r2_score(truth_active, pred_active)
            mse = mean_squared_error(truth_active, pred_active)
            rmse = np.sqrt(mse)
            std_truth = np.std(truth_active)
            std_pred = np.std(pred_active)
            var_ratio = np.var(pred_active) / np.var(truth_active)
            ks_stat, _ = ks_2samp(truth_active, pred_active)
            kld = calculate_kld(truth_active, pred_active)
            
            # Parse Name
            arch = model_name.split('-')[-1]
            training = model_name.split('-')[0]
            
            metrics_results.append({
                'Variable': var_label,
                'Model': training,
                'Arch': arch,
                'Active_R2': r2,
                'RMSE': rmse,
                'Std_Truth': std_truth,
                'Std_Pred': std_pred,
                'Var_Ratio': var_ratio,
                'KS_Stat': ks_stat,
                'KLD': kld,
                'N_Samples': active_count
            })

        # C. Save Data Arrays (If requested)
        if args.save_numpy:
            # Save Full
            full_path = args.output / f"{var}_FULL_DOMAIN.npz"
            np.savez_compressed(full_path, **full_data_dict)
            logger.info(f"  ✓ Saved Full Domain Data ({len(full_truth):,} pts): {full_path.name}")
            
            # Save Active
            active_path = args.output / f"{var}_ACTIVE_FILTERED.npz"
            np.savez_compressed(active_path, **active_data_dict)
            logger.info(f"  ✓ Saved Active Data ({active_count:,} pts): {active_path.name}")

    # 5. Save Metrics CSV
    df = pd.DataFrame(metrics_results)
    df = df.sort_values(by=['Variable', 'Arch', 'Model'])
    
    csv_path = args.output / 'active_turbulence_metrics.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    
    print("\n" + "="*100)
    print(f"METRICS SUMMARY (Threshold > {args.threshold})")
    print("="*100)
    print(df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
    print("="*100)
    print(f"Metrics saved to: {csv_path}")
    if args.save_numpy:
        print(f"Data arrays saved to: {args.output}/*.npz")

if __name__ == "__main__":
    main()
