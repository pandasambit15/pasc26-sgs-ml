# Physics-Aware Multi-Task Learning for Atmospheric Turbulence Parameterization

[![Data & Supplementary](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.20140433-blue)](https://doi.org/10.5281/zenodo.20140433)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3815572.3815750-blue)](https://doi.org/10.1145/3815572.3815750)
[![Conference](https://img.shields.io/badge/PASC-2026-green)](https://pasc26.pasc-conference.org)

Official code for the paper:

> **Physics-Aware Multi-Task Learning for Atmospheric Turbulence Parameterization: Auxiliary Tasks versus Architectural Conditioning**  
> Sambit Kumar Panda, Todd Jones, Muhammad Shahzad, Bryan Lawrence, Anna-Louise Ellis  
> *Platform for Advanced Scientific Computing Conference (PASC '26), June 29–July 01, 2026, Bern, Switzerland*  
> DOI: [10.1145/3815572.3815750](https://doi.org/10.1145/3815572.3815750)

---

## Overview

This repository contains training, experiment, and evaluation scripts for emulating Smagorinsky-based subgrid-scale (SGS) turbulence closures in the [UK Met Office NERC Cloud Model (MONC)](https://github.com/mesham/monc) using physics-aware multi-task learning.

We systematically compare two physics-integration strategies across 54 model configurations:

- **Baseline**: Richardson number prediction as auxiliary gradient regularization
- **Ri-Conditioned**: Explicit architectural conditioning — predicted Ri fed directly into coefficient prediction heads

across three architectures (MLP, ResMLP, TabTransformer) and three task-weighting methods (Manual, Uncertainty Weighting, DWA).

**Key findings:**
- Uncertainty-based task weighting outperforms manual and DWA by 20–30% on coefficient R²
- Simple MLP + Ri-conditioning provides the best cross-regime robustness (+290% on ARM active turbulence)
- Physical constraint compliance is maintained even when predictive accuracy degrades, pointing to data coverage — not physics incompatibility — as the primary failure mode

---

## Repository Structure

```
pasc26-sgs-ml/
├── models/
│   └── ri_conditioned_architectures.py   # RiConditionedMLP, RiConditionedResMLP, RiConditionedTabTransformer
│
├── training/
│   ├── train_baseline_mlp.py             # Baseline MLP training (UnifiedSGSCoefficientNetwork)
│   ├── train_baseline_resmlp.py          # Baseline ResMLP training
│   ├── train_baseline_tabtransformer.py  # Baseline TabTransformer training
│   ├── train_ri_conditioned.py           # Ri-conditioned training (all 3 architectures)
│   ├── task_weight_baseline.py           # Task weight optimization — Baseline (Manual/Uncertainty/GradNorm/DWA)
│   └── task_weight_ri_conditioned.py     # Task weight optimization — Ri-conditioned
│
├── experiments/
│   ├── utils.py                          # Shared utilities (dataset, loss, model factory)
│   ├── l2_ablation_baseline.py           # L2 regularization ablation — Baseline models
│   └── l2_ablation_ri_conditioned.py     # L2 regularization ablation — Ri-conditioned models
│
├── evaluation/
│   ├── comprehensive_analysis.py         # Full diagnostics: spatial, temporal, vertical profiles, scatter
│   ├── constrained_inference.py          # Physical constraint (non-negativity) analysis
│   ├── kde_logspace.py                   # KDE plots in log-space for full domain
│   ├── kde_active_turbulence.py          # KDE plots filtered to active turbulence (coefficient > threshold)
│   ├── active_metrics.py                 # Active turbulence R², RMSE, KLD, Wasserstein + npz export
│   └── extract_physics.py               # Physics quantities time-series extraction from MONC NetCDF files
│
├── configs/
│   └── experiments_config.yaml           # Example configuration file
│
├── data/
│   └── README.md                         # Data description and access instructions
│
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/yourusername/pasc26-sgs-ml.git
cd pasc26-sgs-ml
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.1, CUDA 11.8, on NVIDIA A100 (40 GB).

---

## Usage

### 1. Prepare Data

See `data/README.md` for the expected numpy array format and scaler conventions.

### 2. Train Baseline Models

```bash
# MLP
python training/train_baseline_mlp.py \
    --data-dir /path/to/data \
    --scaler-dir /path/to/scalers \
    --checkpoint-dir checkpoints/baseline_mlp

# ResMLP
python training/train_baseline_resmlp.py \
    --data-dir /path/to/data \
    --scaler-dir /path/to/scalers \
    --checkpoint-dir checkpoints/baseline_resmlp

# TabTransformer
python training/train_baseline_tabtransformer.py \
    --data-dir /path/to/data \
    --scaler-dir /path/to/scalers \
    --checkpoint-dir checkpoints/baseline_tabt
```

### 3. Train Ri-Conditioned Models

```bash
python training/train_ri_conditioned.py \
    --data-dir /path/to/data \
    --scaler-dir /path/to/scalers \
    --architecture MLP \          # MLP | ResMLP | TabTransformer
    --checkpoint-dir checkpoints/ri_mlp
```

### 4. Task Weight Optimization (Main Experiment — Table 4 in paper)

```bash
# Baseline physics approach
python training/task_weight_baseline.py \
    --config configs/experiments_config.yaml \
    --architecture MLP \
    --methods Manual Uncertainty DWA \
    --output-dir results/task_weights/baseline_mlp

# Ri-conditioned physics approach
python training/task_weight_ri_conditioned.py \
    --config configs/experiments_config.yaml \
    --architecture MLP \
    --methods Manual Uncertainty DWA \
    --output-dir results/task_weights/ri_mlp
```

### 5. L2 Regularization Ablation

```bash
python experiments/l2_ablation_baseline.py \
    --config configs/experiments_config.yaml

python experiments/l2_ablation_ri_conditioned.py \
    --config configs/experiments_config.yaml \
    --architecture MLP
```

### 6. Evaluation

```bash
# Full diagnostic suite (vertical profiles, temporal R², scatter)
python evaluation/comprehensive_analysis.py \
    --mode timeseries \
    --data-dir /path/to/netcdf_files \
    --baseline-mlp checkpoints/baseline_mlp/best.pth \
    --ri-mlp checkpoints/ri_mlp/best.pth \
    --scaler-dir /path/to/scalers \
    --output results/full_analysis \
    --k-min 0 --k-max 219

# Physical constraint compliance (Table 8 in paper)
python evaluation/constrained_inference.py \
    --mode timeseries \
    --data-dir /path/to/netcdf_files \
    --baseline-mlp checkpoints/baseline_mlp/best.pth \
    --ri-mlp checkpoints/ri_mlp/best.pth \
    --scaler-dir /path/to/scalers \
    --output results/constraints

# Active turbulence KDE distributions (Figures 5–7 in paper)
python evaluation/kde_active_turbulence.py \
    --data-dir /path/to/netcdf_files \
    --scaler-dir /path/to/scalers \
    --baseline-mlp checkpoints/baseline_mlp/best.pth \
    --ri-mlp checkpoints/ri_mlp/best.pth \
    --threshold 0.01 \
    --output results/kde_active

# Active turbulence metrics (Table 7 in paper)
python evaluation/active_metrics.py \
    --data-dir /path/to/netcdf_files \
    --scaler-dir /path/to/scalers \
    --baseline-mlp checkpoints/baseline_mlp/best.pth \
    --threshold 0.01 \
    --output results/active_metrics
```

---

## Inference Engine

The evaluation scripts depend on two core inference modules included in `evaluation/`:

| Module | Provides |
|---|---|
| `evaluation/inference_engine.py` | `UnifiedInferenceEngine` — batched NetCDF inference for all 6 model variants |
| `evaluation/analysis_pipeline.py` | `FastDataLoader`, `FastFeatureExtractor`, `extract_truth_from_netcdf`, metric utilities |

---

## Experimental Results

| Model | RCE Visc R² | ARM Visc R² | ARM Active R² | Constraint Violations |
|---|---|---|---|---|
| Baseline-MLP | 0.801 | 0.51 | 0.096 | 0.01% |
| Ri-MLP | 0.793 | 0.57 | 0.387 | 0.01% |
| Baseline-ResMLP | 0.782 | 0.33 | −2.27 | 3.82% |
| Ri-ResMLP | 0.771 | 0.14 | −9.08 | 3.89% |
| Baseline-TabT | 0.713 | −0.18 | — | — |
| Ri-TabT | 0.704 | 0.10 | — | — |

Full results in Tables 4–8 of the paper.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{panda2026physics,
  author    = {Panda, Sambit Kumar and Jones, Todd and Shahzad, Muhammad and Lawrence, Bryan and Ellis, Anna-Louise},
  title     = {Physics-Aware Multi-Task Learning for Atmospheric Turbulence Parameterization: Auxiliary Tasks versus Architectural Conditioning},
  booktitle = {Platform for Advanced Scientific Computing Conference (PASC '26)},
  year      = {2026},
  month     = {June},
  address   = {Bern, Switzerland},
  doi       = {10.1145/3815572.3815750},
  publisher = {ACM}
}
```

---

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0), consistent with the paper's open-access license.

---

## Acknowledgements

This research was conducted at the University of Reading with computational resources from RACC2 and the JASMIN supercomputing facility. We acknowledge the UK Met Office for their CASE studentship award, MONC code development support, simulation data access, and HPC resource provision. This work was supported by the University of Reading AFESP Programme [award number A3720300].
