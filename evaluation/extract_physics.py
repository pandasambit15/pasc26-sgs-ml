#!/usr/bin/env python3
"""
Physics Quantities Time Series Extraction - Matching RCE Format
================================================================

Extracts physics quantities matching the exact format of physics_timeseries_rce.json

Variable names from MONC NetCDF files:
- Velocities: u, v, w
- Temperature: th (potential temperature)
- Moisture: q_vapour, q_cloud_liquid_mass
- Richardson number: ri_smag
- Coordinates: z, zn

Usage:
    python extract_physics_quantities.py \
        --data-dir /path/to/netcdf_files/ \
        --output physics_timeseries.json \
        --time-idx 0 \
        --k-min 0 --k-max 219
"""

import numpy as np
import netCDF4 as nc
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional
import argparse
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ===================================================================
#  TIMESTAMP EXTRACTION
# ===================================================================

def extract_timestamp_from_filename(filename: Path) -> Optional[float]:
    """Extract simulation time (seconds) from NetCDF filename."""
    name = filename.stem
    match = re.search(r'(\d+)$', name)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


# ===================================================================
#  NETCDF DATA EXTRACTION
# ===================================================================

def extract_fields_from_netcdf(nc_file: Path, time_idx: int = 0, 
                               k_min: int = 0, k_max: int = 219) -> Dict:
    """
    Extract required fields from MONC NetCDF file.
    
    Variable names in MONC:
    - u, v: Horizontal velocities on zn grid
    - w: Vertical velocity on z grid
    - th: Potential temperature on zn grid
    - q_vapour: Water vapor mixing ratio on zn grid
    - q_cloud_liquid_mass: Cloud liquid water on zn grid
    - ri_smag: Richardson number on zn grid
    
    Returns dictionary with fields.
    """
    with nc.Dataset(nc_file, 'r') as dataset:
        fields = {}
        
        # Get time dimension name (varies by file)
        time_dims = [d for d in dataset.dimensions.keys() if 'time_series' in d]
        if not time_dims:
            raise ValueError(f"No time dimension found in {nc_file.name}")
        time_dim = time_dims[0]
        
        # Velocity components (m/s)
        # u, v are on zn grid
        if 'u' in dataset.variables:
            fields['u'] = dataset.variables['u'][time_idx, :, :, k_min:k_max+1].T
        if 'v' in dataset.variables:
            fields['v'] = dataset.variables['v'][time_idx, :, :, k_min:k_max+1].T
        
        # w is on z grid
        if 'w' in dataset.variables:
            fields['w'] = dataset.variables['w'][time_idx, :, :, k_min:k_max+1].T
        
        # Potential temperature (K) - on zn grid
        if 'th' in dataset.variables:
            fields['theta'] = dataset.variables['th'][time_idx, :, :, k_min:k_max+1].T
        
        # Water vapor mixing ratio (kg/kg) - on zn grid
        if 'q_vapour' in dataset.variables:
            fields['qv'] = dataset.variables['q_vapour'][time_idx, :, :, k_min:k_max+1].T
        
        # Cloud liquid water mixing ratio (kg/kg) - on zn grid
        if 'q_cloud_liquid_mass' in dataset.variables:
            fields['qcl'] = dataset.variables['q_cloud_liquid_mass'][time_idx, :, :, k_min:k_max+1].T
        
        # Richardson number - on zn grid
        if 'ri_smag' in dataset.variables:
            fields['richardson'] = dataset.variables['ri_smag'][time_idx, :, :, k_min:k_max+1].T
        
    return fields


# ===================================================================
#  PHYSICS QUANTITIES CALCULATION
# ===================================================================

def calculate_physics_quantities(fields: Dict) -> Dict:
    """
    Calculate all physics quantities matching the RCE JSON format.
    
    Returns
    -------
    physics : Dict
        Dictionary with all physics quantities
    """
    u = fields['u']
    v = fields['v']
    w = fields['w']
    theta = fields['theta']
    qv = fields.get('qv', np.zeros_like(theta))
    qcl = fields.get('qcl', np.zeros_like(theta))
    richardson = fields.get('richardson', None)
    
    physics = {}
    
    # ========== VELOCITY STATISTICS ==========
    physics['u_mean'] = float(np.nanmean(u))
    physics['u_std'] = float(np.nanstd(u))
    physics['v_mean'] = float(np.nanmean(v))
    physics['v_std'] = float(np.nanstd(v))
    physics['w_mean'] = float(np.nanmean(w))
    physics['w_std'] = float(np.nanstd(w))
    physics['w_max'] = float(np.nanmax(np.abs(w)))
    
    # ========== TEMPERATURE STATISTICS ==========
    physics['theta_mean'] = float(np.nanmean(theta))
    physics['theta_std'] = float(np.nanstd(theta))
    
    # ========== MOISTURE STATISTICS ==========
    physics['qv_mean'] = float(np.nanmean(qv))
    physics['qcl_mean'] = float(np.nanmean(qcl))
    
    # ========== WIND SPEED (CALCULATED) ==========
    # Wind speed = sqrt(u_mean² + v_mean²)
    physics['wind_speed'] = float(np.sqrt(physics['u_mean']**2 + physics['v_mean']**2))
    
    # ========== TURBULENT INTENSITY (CALCULATED) ==========
    # Turbulent intensity = sqrt(σ_u² + σ_v² + σ_w²)
    # This is the total RMS velocity fluctuation
    physics['turbulent_intensity'] = float(np.sqrt(
        physics['u_std']**2 + 
        physics['v_std']**2 + 
        physics['w_std']**2
    ))
    
    # ========== RICHARDSON NUMBER (EXTRACTED FROM DATA) ==========
    if richardson is not None:
        # Remove NaN and Inf values for statistics
        ri_valid = richardson[np.isfinite(richardson)]
        
        if len(ri_valid) > 0:
            physics['ri_mean'] = float(np.mean(ri_valid))
            physics['ri_std'] = float(np.std(ri_valid))
            physics['ri_min'] = float(np.min(ri_valid))
            physics['ri_max'] = float(np.max(ri_valid))
            
            # ========== STABILITY FRACTIONS ==========
            # Unstable: Ri < 0 (buoyancy-driven turbulence)
            # Stable: 0 <= Ri < 0.25 (shear-driven turbulence)
            # Supercritical: Ri >= 0.25 (turbulence suppressed)
            total_points = len(ri_valid)
            
            frac_unstable = np.sum(ri_valid < 0) / total_points
            frac_stable = np.sum((ri_valid >= 0) & (ri_valid < 0.25)) / total_points
            frac_supercritical = np.sum(ri_valid >= 0.25) / total_points
            
            physics['frac_unstable'] = float(frac_unstable)
            physics['frac_stable'] = float(frac_stable)
            physics['frac_supercritical'] = float(frac_supercritical)
        else:
            # No valid Richardson numbers
            physics['ri_mean'] = float('nan')
            physics['ri_std'] = float('nan')
            physics['ri_min'] = float('nan')
            physics['ri_max'] = float('nan')
            physics['frac_unstable'] = 0.0
            physics['frac_stable'] = 0.0
            physics['frac_supercritical'] = 0.0
    else:
        # Richardson number not in dataset
        logger.warning("  ⚠ Richardson number (ri_smag) not found in dataset")
        physics['ri_mean'] = float('nan')
        physics['ri_std'] = float('nan')
        physics['ri_min'] = float('nan')
        physics['ri_max'] = float('nan')
        physics['frac_unstable'] = 0.0
        physics['frac_stable'] = 0.0
        physics['frac_supercritical'] = 0.0
    
    return physics


# ===================================================================
#  TIME SERIES EXTRACTION
# ===================================================================

def extract_timeseries(nc_files: List[Path], time_idx: int, 
                      k_min: int, k_max: int) -> List[Dict]:
    """
    Extract time series of physics quantities from all NetCDF files.
    
    Returns
    -------
    timeseries : List[Dict]
        List of dictionaries matching the RCE JSON format
    """
    timeseries = []
    
    logger.info(f"\nExtracting physics quantities from {len(nc_files)} files...")
    logger.info(f"Domain: k={k_min} to k={k_max} ({k_max-k_min+1} levels)")
    logger.info(f"Time index: {time_idx}\n")
    
    # Check first file for available variables
    logger.info("Checking available variables in first file...")
    with nc.Dataset(nc_files[0], 'r') as ds:
        available_vars = list(ds.variables.keys())
        logger.info(f"  Variables found: {', '.join(available_vars[:20])}...")
        
        # Check for required variables
        required = ['u', 'v', 'w', 'th', 'q_vapour', 'q_cloud_liquid_mass', 'ri_smag']
        missing = [v for v in required if v not in available_vars]
        present = [v for v in required if v in available_vars]
        
        logger.info(f"  ✓ Present: {', '.join(present)}")
        if missing:
            logger.warning(f"  ⚠ Missing: {', '.join(missing)}")
    
    logger.info("")
    
    for i, nc_file in enumerate(nc_files):
        timestamp = extract_timestamp_from_filename(nc_file)
        
        logger.info(f"[{i+1}/{len(nc_files)}] {nc_file.name} | t={timestamp:.0f}s")
        
        try:
            # Extract fields
            fields = extract_fields_from_netcdf(nc_file, time_idx, k_min, k_max)
            
            # Check required fields
            required_fields = ['u', 'v', 'w', 'theta']
            missing_fields = [f for f in required_fields if f not in fields]
            
            if missing_fields:
                logger.warning(f"  ⚠ Missing required fields: {missing_fields}")
                continue
            
            # Calculate physics quantities
            physics = calculate_physics_quantities(fields)
            
            # Create entry matching RCE format
            entry = {
                'file': nc_file.name,
                'timestamp': int(timestamp) if timestamp is not None else i,
                'physics': physics
            }
            
            timeseries.append(entry)
            
            # Log key values
            ri_str = f"{physics['ri_mean']:.1f}" if not np.isnan(physics['ri_mean']) else "N/A"
            logger.info(f"    TI={physics['turbulent_intensity']:.3f}, "
                       f"Ri_mean={ri_str}, "
                       f"unstable={physics['frac_unstable']:.3f}")
            
        except Exception as e:
            logger.error(f"  ✗ Error processing {nc_file.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    return timeseries


# ===================================================================
#  MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract physics quantities time series from MONC NetCDF files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # RCE data
  python extract_physics_quantities.py \\
      --data-dir /path/to/RCE_files/ \\
      --output physics_timeseries_rce.json \\
      --time-idx 0 --k-min 0 --k-max 98

  # ARM data
  python extract_physics_quantities.py \\
      --data-dir /path/to/ARM_files/ \\
      --output physics_timeseries_arm.json \\
      --time-idx 0 --k-min 0 --k-max 219
        """
    )
    
    parser.add_argument('--data-dir', type=Path, required=True,
                       help='Directory containing NetCDF files')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output JSON file')
    parser.add_argument('--time-idx', type=int, default=0,
                       help='Time index within each NetCDF file (default: 0)')
    parser.add_argument('--k-min', type=int, default=0,
                       help='Minimum vertical level (default: 0)')
    parser.add_argument('--k-max', type=int, default=219,
                       help='Maximum vertical level (default: 219)')
    
    args = parser.parse_args()
    
    logger.info(f"\n{'='*70}")
    logger.info("PHYSICS QUANTITIES TIME SERIES EXTRACTION")
    logger.info(f"{'='*70}\n")
    
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Vertical domain: k={args.k_min} to k={args.k_max}")
    logger.info(f"Time index: {args.time_idx}")
    
    # Get NetCDF files
    nc_files = sorted(
        args.data_dir.glob('*.nc'),
        key=lambda p: int(re.search(r'(\d+)\.nc$', p.name).group(1)) 
        if re.search(r'(\d+)\.nc$', p.name) else 0
    )
    
    if not nc_files:
        logger.error(f"\n✗ No NetCDF files found in {args.data_dir}")
        return
    
    logger.info(f"\nFound {len(nc_files)} NetCDF files")
    logger.info(f"  First: {nc_files[0].name}")
    logger.info(f"  Last:  {nc_files[-1].name}\n")
    
    # Extract time series
    timeseries = extract_timeseries(nc_files, args.time_idx, args.k_min, args.k_max)
    
    if not timeseries:
        logger.error("\n✗ No data extracted")
        return
    
    # Save to JSON
    logger.info(f"\n{'='*70}")
    logger.info("SAVING RESULTS")
    logger.info(f"{'='*70}\n")
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'w') as f:
        json.dump(timeseries, f, indent=2)
    
    logger.info(f"✓ Saved: {args.output}")
    logger.info(f"  Entries: {len(timeseries)}")
    
    # Summary statistics
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY STATISTICS")
    logger.info(f"{'='*70}\n")
    
    # Extract time range
    timestamps = [entry['timestamp'] for entry in timeseries]
    logger.info(f"Time range: {min(timestamps)} to {max(timestamps)} seconds")
    logger.info(f"Duration: {(max(timestamps) - min(timestamps))/3600:.2f} hours")
    
    # Average values
    ti_values = [entry['physics']['turbulent_intensity'] for entry in timeseries]
    ri_mean_values = [entry['physics']['ri_mean'] for entry in timeseries 
                     if not np.isnan(entry['physics']['ri_mean'])]
    frac_unstable_values = [entry['physics']['frac_unstable'] for entry in timeseries]
    
    logger.info(f"\nTurbulent Intensity:")
    logger.info(f"  Range: {min(ti_values):.3f} to {max(ti_values):.3f}")
    logger.info(f"  Mean:  {np.mean(ti_values):.3f}")
    
    if ri_mean_values:
        logger.info(f"\nRichardson Number:")
        logger.info(f"  Mean Ri range: {min(ri_mean_values):.1f} to {max(ri_mean_values):.1f}")
        logger.info(f"  Time-averaged mean Ri: {np.mean(ri_mean_values):.1f}")
    else:
        logger.warning(f"\nRichardson Number: Not available in data")
    
    logger.info(f"\nStability:")
    logger.info(f"  Unstable fraction: {np.mean(frac_unstable_values):.3f}")
    logger.info(f"  Stable fraction: {np.mean([e['physics']['frac_stable'] for e in timeseries]):.3f}")
    logger.info(f"  Supercritical fraction: {np.mean([e['physics']['frac_supercritical'] for e in timeseries]):.3f}")
    
    logger.info(f"\n✅ Extraction complete!\n")


if __name__ == '__main__':
    main()
