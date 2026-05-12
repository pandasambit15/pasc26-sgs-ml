# Data

The MONC simulation data used in this study is **not included** in this repository due to size constraints.

## Data Description

Two atmospheric simulation regimes were used:

| Dataset | Grid | Δx | Files | Description |
|---|---|---|---|---|
| RCE | 64×64×99 | ~1000 m | 27 | Radiative Convective Equilibrium — deep tropical convection |
| ARM | 192×192×220 | ~50 m | 9 | Atmospheric Radiation Measurement — shallow mid-latitude cumulus |

Data was generated using the [Met Office NERC Cloud Model (MONC)](https://github.com/mesham/monc).

## Processed Numpy Arrays

The training scripts expect pre-processed `.npy` files and `.pkl` scalers:

```
data_dir/
├── features.npy       # (N, 54) float32 — input feature matrix
├── visc_coeff.npy     # (N,) float32 — eddy viscosity Km (m²/s)
├── diff_coeff.npy     # (N,) float32 — eddy diffusivity Kh (m²/s)
├── richardson.npy     # (N,) float32 — Richardson number Ri
└── regime.npy         # (N,) int64   — stability regime label (0/1/2)

scalers/
├── feature_scaler.pkl
├── visc_scaler.pkl
├── diff_scaler.pkl
└── richardson_scaler.pkl
```

All scalers are `sklearn.preprocessing.RobustScaler` (fitted on training split only).

## Feature Engineering

The 54-dimensional input vector comprises:

| Group | Features | Description |
|---|---|---|
| Prognostic variables | 6 | u, v, w, θ, q_v, q_c at grid point |
| Local stencil | 36 | 6-point star stencil (i±1, j±1, k±1) × 6 variables |
| Grid metadata | 5 | Δx, Δy, Δz, z, θ_ref |
| Geometry | 7 | z/z_top, absolute height, distance to top, x/Lx, y/Ly, boundary distances |

## Data Access

For data access, please contact the corresponding author:
- **Sambit Kumar Panda** — s.panda@pgr.reading.ac.uk  
  University of Reading, UK
