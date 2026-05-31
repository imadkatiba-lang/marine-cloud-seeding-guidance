#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Marine Cloud Seeding Guidance Pipeline
=======================================
End-to-end decision-support framework for auditable marine cloud seeding
over the Moroccan Atlantic offshore domain.

Stages
------
0  : ERA5 data loading, ocean mask construction, and field visualization
A  : Baseline-anchored precipitation predictor (AR-residual, DeltaNet)
B  : Reinforcement decision module (MS-DTAC-6A, actor–critic)
C  : Raw action field visualization (ocean-only sanity check)
D  : Physically gated three-mode mapping (hygroscopic / glaciogenic / dynamic)
E  : Technique-level interpretability with delta-P linkage

Reference
---------
Katiba, I., Belmajdoub, H., Minaoui, K. (2025).
"Auditable Marine Cloud Seeding Guidance via Physically Gated Reinforcement
Decisions and Three-Mode Mapping."

Requirements
------------
numpy, pandas, torch, scikit-learn, xarray, cartopy, matplotlib
"""

import os
import json
import glob
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


# =============================================================================
# Configuration
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The default paths reproduce the local experimental setup used for the manuscript.
# For GitHub usage, override them without editing the script:
#   Windows PowerShell:
#       $env:MARINE_BASE_IN="C:\\path\\to\\Marine"
#       $env:MARINE_OUTDIR="C:\\path\\to\\outputs"
#   Linux/macOS:
#       export MARINE_BASE_IN="/path/to/Marine"
#       export MARINE_OUTDIR="/path/to/outputs"
BASE_IN = os.environ.get(
    "MARINE_BASE_IN",
    r"C:\Users\tuf-p\Desktop\ARTICLES\Marine",
)
OUTDIR = os.environ.get(
    "MARINE_OUTDIR",
    os.path.join(BASE_IN, "marine_cloud_seeding_outputs"),
)

FIGDIR   = os.path.join(OUTDIR, "figs")
CSVDIR   = os.path.join(OUTDIR, "csv")
PTDIR    = os.path.join(OUTDIR, "models")
NPYDIR   = os.path.join(OUTDIR, "npy")
RESDIR   = os.path.join(OUTDIR, "results", "Final-INTER", "analysis_interpretability_v3")
INTERCSV = os.path.join(RESDIR, "csv")
INTERFIG = os.path.join(RESDIR, "figs")

for _dir in [OUTDIR, FIGDIR, CSVDIR, PTDIR, NPYDIR, RESDIR, INTERCSV, INTERFIG]:
    os.makedirs(_dir, exist_ok=True)

DPI_SAVE    = 400
CMAP_VARS   = "cividis"
CMAP_SCORES = "magma"
RANDOM_SEED = int(os.environ.get("MARINE_RANDOM_SEED", "42"))


def set_global_seed(seed: int = 42) -> None:
    """Set NumPy, Python, and PyTorch random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_global_seed(RANDOM_SEED)

print(f"Device : {DEVICE}")
print(f"Input  : {BASE_IN}")
print(f"Output : {OUTDIR}")


# =============================================================================
# Utility functions
# =============================================================================

def add_lonlat_labels(ax, proj=ccrs.PlateCarree(), fontsize: int = 9):
    """Add formatted longitude / latitude gridlines to a Cartopy axes."""
    gl = ax.gridlines(
        crs=proj, draw_labels=True,
        linewidth=0.6, alpha=0.35,
        linestyle="--", color="black",
        x_inline=False, y_inline=False,
    )
    gl.top_labels   = False
    gl.right_labels = False
    gl.xformatter   = LONGITUDE_FORMATTER
    gl.yformatter   = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": fontsize}
    gl.ylabel_style = {"size": fontsize}
    return gl


def open_xr_robust(path: str) -> xr.Dataset:
    """Open a NetCDF dataset, trying multiple engines for compatibility."""
    for engine in ("netcdf4", "h5netcdf", "scipy"):
        try:
            return xr.open_dataset(path, engine=engine, cache=False)
        except Exception:
            pass
    return xr.open_dataset(path, cache=False)


def display_prepare(Z: np.ndarray, lats_1d: np.ndarray, lons_1d: np.ndarray):
    """
    Ensure that latitude increases southward→northward and longitude
    increases westward→eastward so that imshow renders correctly.

    Returns
    -------
    Z_sorted : np.ndarray
    extent   : list[float]  [lon_min, lon_max, lat_min, lat_max]
    """
    Z    = np.array(Z, dtype=float)
    lats = np.array(lats_1d, dtype=float).copy()
    lons = np.array(lons_1d, dtype=float).copy()

    if lats[0] > lats[-1]:
        lats = lats[::-1]
        Z    = Z[::-1, :]

    if lons[0] > lons[-1]:
        lons = lons[::-1]
        Z    = Z[:, ::-1]

    extent = [float(lons.min()), float(lons.max()),
              float(lats.min()), float(lats.max())]
    return Z, extent


def plot_smooth_field(ax, field: np.ndarray, title: str,
                      proj, extent, vmin=None, vmax=None, cmap: str = "viridis"):
    """
    Render a 2-D field with bicubic interpolation, land overlay, and
    lon/lat gridline labels.
    """
    ax.set_extent(extent, crs=proj)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(alpha=0.0)

    masked = np.ma.masked_invalid(np.array(field, dtype=float))
    im = ax.imshow(
        masked, origin="lower", extent=extent, transform=proj,
        interpolation="bicubic", cmap=cmap_obj,
        vmin=vmin, vmax=vmax, zorder=1,
    )
    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="none", zorder=10)
    ax.coastlines(resolution="10m", linewidth=0.8, zorder=11)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    add_lonlat_labels(ax, proj=proj, fontsize=9)
    return im


def dilate_mask(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    """Binary dilation of a boolean mask using 8-connectivity."""
    m = mask.astype(bool)
    for _ in range(int(iters)):
        m = (
            m
            | np.roll(m,  1, 0) | np.roll(m, -1, 0)
            | np.roll(m,  1, 1) | np.roll(m, -1, 1)
            | np.roll(np.roll(m,  1, 0),  1, 1)
            | np.roll(np.roll(m,  1, 0), -1, 1)
            | np.roll(np.roll(m, -1, 0),  1, 1)
            | np.roll(np.roll(m, -1, 0), -1, 1)
        )
    return m


def fill_nans_for_display(Z: np.ndarray, valid_mask: np.ndarray,
                           iters: int = 3) -> np.ndarray:
    """
    Fill NaN values inside the valid region by iterative neighbor averaging.
    Intended for visualization only; does not modify the decision data.
    """
    Z  = np.array(Z, dtype=float, copy=True)
    vm = valid_mask.astype(bool)
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for _ in range(int(iters)):
        nan_in_valid = (~np.isfinite(Z)) & vm
        if not nan_in_valid.any():
            break
        neighbors = np.stack(
            [np.roll(np.roll(Z, dy, axis=0), dx, axis=1) for dy, dx in shifts]
        )
        neighbor_mean = np.nanmean(neighbors, axis=0)
        update = nan_in_valid & np.isfinite(neighbor_mean)
        if not update.any():
            break
        Z[update] = neighbor_mean[update]
    return Z


def nearest_idx(vec: np.ndarray, x: float) -> int:
    """Return the index of the element in `vec` closest to `x`."""
    return int(np.argmin(np.abs(vec - x)))


def save_json(path: str, obj: dict) -> None:
    """Serialize `obj` to a JSON file with readable indentation."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    print(f"Saved : {path}")


def safe_num(series):
    """Coerce a pandas Series to numeric, setting non-parseable values to NaN."""
    return pd.to_numeric(series, errors="coerce")


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Element-wise logistic sigmoid."""
    x = np.array(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def require_file(path: str, label: str = "file") -> str:
    """Fail early with a readable error message if a required input is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required {label} not found: {path}")
    return path


# =============================================================================
# Stage 0 — ERA5 Loading, Ocean Mask, and Field Visualization
# =============================================================================
print("\n[Stage 0] ERA5 loading and ocean mask construction")

ERA5_DIR     = os.path.join(BASE_IN, "ERA5_Marine_2025_01")
INSTANT_FILE = os.path.join(ERA5_DIR, "data_stream-oper_stepType-instant.nc")
AVG_FILE     = os.path.join(ERA5_DIR, "data_stream-oper_stepType-avg.nc")
ACCUM_FILE   = os.path.join(ERA5_DIR, "data_stream-oper_stepType-accum.nc")

ds_instant = open_xr_robust(require_file(INSTANT_FILE, 'instant ERA5 stream'))
ds_avg     = open_xr_robust(require_file(AVG_FILE, 'averaged ERA5 stream'))
ds_accum   = open_xr_robust(require_file(ACCUM_FILE, 'accumulated ERA5 stream'))

# Identify time dimension
time_dim = None
for _candidate in ("valid_time", "time"):
    if _candidate in ds_instant.dims:
        time_dim = _candidate
        break
if time_dim is None:
    raise ValueError("No time dimension found in the instant stream (expected 'valid_time' or 'time').")

T0 = 0  # Reference time index for mask and covariate extraction

def to_2d(da):
    """Extract a single 2-D slice at the reference time step."""
    return da.isel({time_dim: T0}).values

lat_name = "latitude" if "latitude" in ds_instant.coords else "lat"
lon_name = "longitude" if "longitude" in ds_instant.coords else "lon"

lats_era = ds_instant[lat_name].values
lons_era = ds_instant[lon_name].values

if "sst" not in ds_instant:
    raise ValueError("Variable 'sst' not found in the instant stream.")

sst         = ds_instant["sst"].isel({time_dim: T0})
ocean_mask_era  = np.isfinite(sst.values)
ocean_ratio = float(np.nanmean(ocean_mask_era.astype(float)))
print(f"Ocean fraction (finite SST): {ocean_ratio:.4f}")

# Dilated mask used exclusively for near-coast visualization
OCEAN_DILATE    = 2
ocean_mask_plot = dilate_mask(ocean_mask_era, iters=OCEAN_DILATE)

def mask_ocean_plot(Z: np.ndarray) -> np.ndarray:
    """Set land pixels to NaN using the dilated ocean mask (visualization only)."""
    Z = np.array(Z, dtype=float)
    Z[~ocean_mask_plot] = np.nan
    return Z

# ── Field overview plot ──────────────────────────────────────────────────────
print("[Stage 0] Plotting ERA5 variable overview...")

var_list = [v for v in ["sst", "u10", "v10", "t2m", "d2m", "sp", "blh", "cape"]
            if v in ds_instant]
ncols = 3
nrows = int(np.ceil(len(var_list) / ncols))

proj = ccrs.PlateCarree()
fig, axs = plt.subplots(
    nrows, ncols, figsize=(14, 4 * nrows),
    subplot_kw={"projection": proj},
)
axs = np.atleast_1d(axs).ravel()

cmap_vars_obj = plt.get_cmap(CMAP_VARS).copy()
cmap_vars_obj.set_bad(alpha=0.0)

for i, v in enumerate(var_list):
    ax = axs[i]
    Z  = mask_ocean_plot(to_2d(ds_instant[v]))
    if v == "sst":
        Z = fill_nans_for_display(Z, valid_mask=ocean_mask_plot, iters=3)

    Z_disp, extent = display_prepare(Z, lats_era, lons_era)
    Z_masked = np.ma.masked_invalid(Z_disp)

    ax.set_extent(extent, crs=proj)
    im = ax.imshow(
        Z_masked, origin="lower", extent=extent, transform=proj,
        interpolation="bicubic", cmap=cmap_vars_obj, zorder=1,
    )
    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="none", zorder=10)
    ax.coastlines(resolution="10m", linewidth=0.8, zorder=11)
    ax.set_title(v, fontsize=11)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    add_lonlat_labels(ax, proj=proj, fontsize=9)
    cb = plt.colorbar(im, ax=ax, orientation="vertical",
                      fraction=0.046, pad=0.2, location="left")
    cb.set_label(v, fontsize=9)

for j in range(len(var_list), len(axs)):
    axs[j].axis("off")

plt.tight_layout()
_path = os.path.join(FIGDIR, "stage0_era5_variable_overview.png")
plt.savefig(_path, dpi=DPI_SAVE, bbox_inches="tight")
plt.show()
print(f"Saved : {_path}")


# =============================================================================
# Stage A — Baseline-Anchored Precipitation Predictor (DeltaNet)
# =============================================================================
print("\n[Stage A] AR-residual precipitation predictor")

STATE_PATH = os.path.join(BASE_IN, "STATE_512.npy")
TP_PATH    = os.path.join(BASE_IN, "TP_TARGET.npy")

STATE = np.load(require_file(STATE_PATH, 'state tensor')).astype(np.float32)   # shape (T, 512)
TP    = np.load(require_file(TP_PATH, 'precipitation target')).astype(np.float32)      # shape (T,)

TP    = np.maximum(TP, 0.0)
STATE[~np.isfinite(STATE)] = 0.0
TP[~np.isfinite(TP)]       = 0.0

# ── Autoregressive feature construction ─────────────────────────────────────
LAGS = 2
if len(TP) <= LAGS + 5:
    raise ValueError(f"Insufficient samples for LAGS={LAGS}.")

Y_raw  = TP[LAGS:]
L1_raw = TP[LAGS - 1:-1]
L2_raw = TP[LAGS - 2:-2]
S_now  = STATE[LAGS:]

# Log-transform stabilization: z = log(1 + y)
Y_log  = np.log1p(Y_raw).astype(np.float32)
L1_log = np.log1p(L1_raw).astype(np.float32)
L2_log = np.log1p(L2_raw).astype(np.float32)

X_state_cpu = torch.from_numpy(S_now).float()
l1_cpu      = torch.from_numpy(L1_log).float().unsqueeze(-1)
l2_cpu      = torch.from_numpy(L2_log).float().unsqueeze(-1)
ylog_cpu    = torch.from_numpy(Y_log).float()

N, D_state = X_state_cpu.shape
X_dim = D_state + 2
print(f"Samples : {N}  |  State dim : {D_state}  |  Input dim : {X_dim}")

# ── Chronological train / validation / test split ───────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15

i_train = int(N * TRAIN_FRAC)
i_val   = int(N * (TRAIN_FRAC + VAL_FRAC))

idx_tr = np.arange(0, i_train)
idx_va = np.arange(i_train, i_val)
idx_te = np.arange(i_val, N)


def rmse_r2(y_true, y_pred):
    """Return RMSE and R² in the original precipitation space."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    return rmse, r2


# Persistence baseline: ŷ_t = y_{t-1}
rmse_btr, r2_btr = rmse_r2(Y_raw[idx_tr], L1_raw[idx_tr])
rmse_bva, r2_bva = rmse_r2(Y_raw[idx_va], L1_raw[idx_va])
rmse_bte, r2_bte = rmse_r2(Y_raw[idx_te], L1_raw[idx_te])
print(
    f"[Persistence] R²  train={r2_btr:.4f}  val={r2_bva:.4f}  test={r2_bte:.4f}"
)
print(
    f"[Persistence] RMSE train={rmse_btr:.4f}  val={rmse_bva:.4f}  test={rmse_bte:.4f}"
)

split_info = {
    "N_total":  int(N),
    "state_dim": int(D_state),
    "lags":     int(LAGS),
    "train_idx": [int(idx_tr[0]), int(idx_tr[-1])],
    "val_idx":   [int(idx_va[0]), int(idx_va[-1])],
    "test_idx":  [int(idx_te[0]), int(idx_te[-1])],
    "baseline_persistence": {
        "rmse_train": rmse_btr, "r2_train": r2_btr,
        "rmse_val":   rmse_bva, "r2_val":   r2_bva,
        "rmse_test":  rmse_bte, "r2_test":  r2_bte,
    },
}
save_json(os.path.join(CSVDIR, "predictor_split_info.json"), split_info)

# ── Robust feature scaling (training partition only) ─────────────────────────
Xtr_state = X_state_cpu[idx_tr]
with torch.no_grad():
    scale_median = Xtr_state.median(dim=0).values
    q1           = torch.quantile(Xtr_state, 0.25, dim=0)
    q3           = torch.quantile(Xtr_state, 0.75, dim=0)
    scale_iqr    = torch.clamp(q3 - q1, min=1e-6)


def scale_state(Xs: torch.Tensor) -> torch.Tensor:
    """Apply training-partition robust scaling to state features."""
    return (Xs - scale_median) / scale_iqr


X_all_cpu     = torch.cat([scale_state(X_state_cpu), l1_cpu, l2_cpu], dim=1)
l1_anchor_cpu = l1_cpu.squeeze(-1)

X_all     = X_all_cpu.to(DEVICE, non_blocking=True)
ylog      = ylog_cpu.to(DEVICE, non_blocking=True)
l1_anchor = l1_anchor_cpu.to(DEVICE, non_blocking=True)

Xtr, Xva, Xte = X_all[idx_tr], X_all[idx_va], X_all[idx_te]
ytr, yva, yte = ylog[idx_tr], ylog[idx_va], ylog[idx_te]
a_tr, a_va, a_te = l1_anchor[idx_tr], l1_anchor[idx_va], l1_anchor[idx_te]


# ── DeltaNet: autoregressive residual predictor ──────────────────────────────
class DeltaNet(nn.Module):
    """
    Compact MLP that learns the additive residual correction
    Δz_t = z_t − z_{t-1}, conditioned on the environmental state.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


model     = DeltaNet(X_dim).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)


@torch.no_grad()
def evaluate(Xs, ylog_true, anchor_log):
    """Compute RMSE and R² in the original precipitation space."""
    model.eval()
    delta = model(Xs)
    z_hat = anchor_log + delta
    y_hat = torch.expm1(torch.clamp(z_hat, min=0.0))
    y_true = torch.expm1(torch.clamp(ylog_true, min=0.0))
    y = y_true.detach().cpu().numpy()
    p = y_hat.detach().cpu().numpy()
    return rmse_r2(y, p)


EPOCHS_PRED = 200
BATCH_SIZE  = 1024
PATIENCE    = 20
best_val    = np.inf
patience_ct = 0
best_state  = None

for epoch in range(1, EPOCHS_PRED + 1):
    model.train()
    perm = torch.randperm(len(idx_tr), device=DEVICE)

    for i in range(0, len(idx_tr), BATCH_SIZE):
        ids = perm[i:i + BATCH_SIZE]
        xb, yb, ab = Xtr[ids], ytr[ids], a_tr[ids]

        optimizer.zero_grad(set_to_none=True)
        delta = model(xb)
        z_hat = ab + delta

        loss_fit = F.smooth_l1_loss(z_hat, yb, beta=0.15)
        loss_reg = 1e-3 * torch.mean(delta ** 2)
        (loss_fit + loss_reg).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    rmse_va, _ = evaluate(Xva, yva, a_va)
    if epoch % 10 == 0:
        rmse_tr, r2_tr = evaluate(Xtr, ytr, a_tr)
        rmse_va, r2_va = evaluate(Xva, yva, a_va)
        print(
            f"  [A] Epoch {epoch:03d} | "
            f"RMSE train={rmse_tr:.4f} R²={r2_tr:.4f} | "
            f"RMSE val={rmse_va:.4f} R²={r2_va:.4f}"
        )

    if rmse_va < best_val - 1e-6:
        best_val    = rmse_va
        patience_ct = 0
        best_state  = {k: v.detach().cpu().clone()
                       for k, v in model.state_dict().items()}
    else:
        patience_ct += 1
        if patience_ct >= PATIENCE:
            print(f"  [A] Early stopping at epoch {epoch}.")
            break

if best_state is not None:
    model.load_state_dict(best_state)

rmse_tr, r2_tr = evaluate(Xtr, ytr, a_tr)
rmse_va, r2_va = evaluate(Xva, yva, a_va)
rmse_te, r2_te = evaluate(Xte, yte, a_te)

final_metrics = {
    "lags": int(LAGS),
    "rmse_train": rmse_tr, "r2_train": r2_tr,
    "rmse_val":   rmse_va, "r2_val":   r2_va,
    "rmse_test":  rmse_te, "r2_test":  r2_te,
    "baseline_persistence": split_info["baseline_persistence"],
}
print(f"[Stage A] Final metrics : {final_metrics}")
save_json(os.path.join(CSVDIR, "predictor_final_metrics.json"), final_metrics)

torch.save(model.state_dict(), os.path.join(PTDIR, "DELTANET.pt"))
print(f"Saved : {os.path.join(PTDIR, 'DELTANET.pt')}")


# =============================================================================
# Stage B — Reinforcement Decision Module (MS-DTAC-6A)
# =============================================================================
print("\n[Stage B] Reinforcement learning — MS-DTAC-6A actor–critic")


def scale_state_device(X: torch.Tensor) -> torch.Tensor:
    """Transfer robust scaling parameters to the current device on demand."""
    md = scale_median.to(X.device)
    iq = scale_iqr.to(X.device)
    n_state = md.numel()
    if X.shape[1] == n_state:
        return (X - md) / iq
    # State + lag features already concatenated
    Xs = (X[:, :n_state] - md) / iq
    return torch.cat([Xs, X[:, n_state:]], dim=1)


# X_all already contains training-only robust-scaled state features concatenated
# with the two log-precipitation lags. Re-scaling it here would double-scale the
# first 512 state dimensions and distort the RL state distribution.
with torch.no_grad():
    X_rl_scaled = X_all

Y_rl   = torch.from_numpy(Y_raw).float().to(DEVICE)
N_rl   = X_rl_scaled.shape[0]
STATE_DIM  = X_rl_scaled.shape[1]
ACTION_DIM = 6


class Actor(nn.Module):
    """Policy network mapping state → bounded continuous action ∈ [−1, 1]^6."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Tanh(),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class Critic(nn.Module):
    """Q-value network estimating Q(s, a)."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))


actor      = Actor(STATE_DIM, ACTION_DIM).to(DEVICE)
critic     = Critic(STATE_DIM, ACTION_DIM).to(DEVICE)
critic_tgt = Critic(STATE_DIM, ACTION_DIM).to(DEVICE)
critic_tgt.load_state_dict(critic.state_dict())

opt_actor  = torch.optim.AdamW(actor.parameters(),  lr=1e-4)
opt_critic = torch.optim.AdamW(critic.parameters(), lr=2e-4)

GAMMA      = 0.95
TAU        = 0.01
BATCH_RL   = 256
EPOCHS_RL  = 300

actions_log = []
q_log       = []
action_l2_log = []

actor.train()
critic.train()

for epoch in range(1, EPOCHS_RL + 1):
    idx = torch.from_numpy(
        np.random.choice(N_rl, BATCH_RL, replace=True)
    ).long().to(DEVICE)

    s  = X_rl_scaled[idx]
    tp = Y_rl[idx].unsqueeze(-1)

    # Actor update: maximise Q(s, π(s))
    a      = actor(s)
    q      = critic(s, a)
    loss_a = -q.mean()
    opt_actor.zero_grad(set_to_none=True)
    loss_a.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    opt_actor.step()

    # Critic update: minimise TD error with target network
    with torch.no_grad():
        a_next  = actor(s)
        q_tgt   = critic_tgt(s, a_next)
        td_tgt  = tp + GAMMA * q_tgt

    q_pred  = critic(s, a_next)
    loss_c  = F.mse_loss(q_pred, td_tgt)
    opt_critic.zero_grad(set_to_none=True)
    loss_c.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    opt_critic.step()

    # Soft target-network update
    with torch.no_grad():
        for p, pt in zip(critic.parameters(), critic_tgt.parameters()):
            pt.data.copy_(TAU * p.data + (1.0 - TAU) * pt.data)

    A_np = a.detach().cpu().numpy()
    actions_log.append(A_np)
    q_log.append(float(q.mean().item()))
    action_l2_log.append(float(np.mean(np.linalg.norm(A_np, axis=1))))

    if epoch % 20 == 0:
        print(
            f"  [B] Epoch {epoch:03d} | "
            f"Q={q_log[-1]:.4f}  ‖a‖₂={action_l2_log[-1]:.4f}"
        )

np.save(os.path.join(NPYDIR, "ACTIONS_LOG.npy"), np.array(actions_log))
print(f"Saved : {os.path.join(NPYDIR, 'ACTIONS_LOG.npy')}")

rl_metrics = pd.DataFrame({
    "epoch":      np.arange(1, EPOCHS_RL + 1),
    "Q_mean":     q_log,
    "action_L2_mean": action_l2_log,
})
rl_metrics.to_csv(os.path.join(CSVDIR, "rl_training_metrics.csv"), index=False)
print(f"Saved : {os.path.join(CSVDIR, 'rl_training_metrics.csv')}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(rl_metrics["epoch"], rl_metrics["Q_mean"])
ax1.set_title("RL training — mean Q-value")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Q mean")

ax2.plot(rl_metrics["epoch"], rl_metrics["action_L2_mean"])
ax2.set_title("RL training — mean action ‖a‖₂")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("‖a‖₂ mean")

plt.tight_layout()
_path = os.path.join(FIGDIR, "stageB_rl_training_diagnostics.png")
plt.savefig(_path, dpi=DPI_SAVE, bbox_inches="tight")
plt.show()
print(f"Saved : {_path}")


# =============================================================================
# Stage C — Raw Action Field Visualization (Sanity Check)
# =============================================================================
print("\n[Stage C] Raw RL action field visualization (ocean-only)")

LAT_PATH = os.path.join(BASE_IN, "LAT_REAL.npy")
LON_PATH = os.path.join(BASE_IN, "LON_REAL.npy")

if not os.path.exists(LAT_PATH):
    raise FileNotFoundError(f"Missing: {LAT_PATH}")
if not os.path.exists(LON_PATH):
    raise FileNotFoundError(f"Missing: {LON_PATH}")

LAT = np.load(require_file(LAT_PATH, "latitude vector"))
LON = np.load(require_file(LON_PATH, "longitude vector"))

A_last = actions_log[-1]          # actions from the final RL epoch
N_RL   = A_last.shape[0]
n_side = int(np.sqrt(N_RL))
if n_side * n_side != N_RL:
    raise ValueError(
        f"RL grid is not square (N_RL={N_RL}). "
        "Adjust the grid construction to match the domain."
    )

lon_axis = np.linspace(float(np.nanmin(LON)), float(np.nanmax(LON)), n_side)
lat_axis = np.linspace(float(np.nanmin(LAT)), float(np.nanmax(LAT)), n_side)
LON2, LAT2 = np.meshgrid(lon_axis, lat_axis)

sst_vals = sst.values


def build_ocean_mask_rl() -> np.ndarray:
    """Build the ocean mask on the RL decision grid via nearest-ERA5-neighbor lookup."""
    mask = np.zeros((n_side, n_side), dtype=bool)
    for iy in range(n_side):
        for ix in range(n_side):
            j = nearest_idx(lats_era, LAT2[iy, ix])
            k = nearest_idx(lons_era, LON2[iy, ix])
            mask[iy, ix] = bool(np.isfinite(sst_vals[j, k]))
    return mask


ocean_mask_rl = build_ocean_mask_rl()
ocean_flat    = ocean_mask_rl.flatten()
LON_flat      = LON2.flatten()
LAT_flat      = LAT2.flatten()

ACTION_LABELS = ["SLP", "Humidity", "U10", "V10", "SST", "SSS"]
proj = ccrs.PlateCarree()

fig, axs = plt.subplots(2, 3, figsize=(16, 8), subplot_kw={"projection": proj})
for i, ax in enumerate(axs.flat):
    vals = np.abs(A_last[:, i]).astype(float)
    vmax = np.nanmax(vals) if np.isfinite(vals).any() else 1.0
    vals = vals / (vmax if vmax > 0 else 1.0)

    sc = ax.scatter(
        LON_flat[ocean_flat], LAT_flat[ocean_flat],
        c=vals[ocean_flat], cmap="viridis", s=50,
        vmin=0.0, vmax=1.0, transform=proj, zorder=1,
    )
    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="none", zorder=10)
    ax.coastlines(resolution="10m", linewidth=0.8, zorder=11)
    ax.set_title(ACTION_LABELS[i], fontsize=11)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    add_lonlat_labels(ax, proj=proj, fontsize=9)
    cb = plt.colorbar(sc, ax=ax, orientation="vertical",
                      fraction=0.046, pad=0.2, location="left")
    cb.set_label("Relative action intensity", fontsize=9)

plt.suptitle("Marine Cloud Seeding — Raw RL Action Fields (Ocean Only)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
_path = os.path.join(FIGDIR, "stageC_raw_actions_scatter.png")
plt.savefig(_path, dpi=DPI_SAVE, bbox_inches="tight")
plt.show()
print(f"Saved : {_path}")


# =============================================================================
# Stage D — Physically Gated Three-Mode Operational Mapping
# =============================================================================
print("\n[Stage D] Physically gated three-mode mapping (hygroscopic / glaciogenic / dynamic)")

THR_INTENSITY = 0.20   # Minimum ‖a‖₂ required for a GO recommendation


def ocean_only(F: np.ndarray) -> np.ndarray:
    """Set non-ocean pixels to NaN on the RL decision grid."""
    F = np.array(F, dtype=float)
    F[~ocean_mask_rl] = np.nan
    return F


def era_to_rl_nearest(Z_era_2d: np.ndarray) -> np.ndarray:
    """
    Resample an ERA5 field to the RL decision grid via nearest-neighbor lookup.
    """
    out = np.full((n_side, n_side), np.nan, dtype=float)
    for iy in range(n_side):
        for ix in range(n_side):
            j = nearest_idx(lats_era, LAT2[iy, ix])
            k = nearest_idx(lons_era, LON2[iy, ix])
            out[iy, ix] = float(Z_era_2d[j, k])
    return out


def normalize_by_quantile(F: np.ndarray, q_hi: float = 0.98) -> np.ndarray:
    """Scale a field to [0, 1] using a high-quantile ceiling."""
    F   = np.array(F, dtype=float)
    x   = F[np.isfinite(F)]
    if x.size == 0:
        return F
    hi  = float(np.quantile(x, q_hi))
    if not np.isfinite(hi) or hi <= 1e-12:
        return np.clip(F, 0.0, 1.0)
    return np.clip(F / hi, 0.0, 1.0)


# ── Action decomposition ─────────────────────────────────────────────────────
A2      = A_last.reshape(n_side, n_side, ACTION_DIM)
A_abs   = np.abs(A2)
A_slp   = A_abs[:, :, 0]
A_hum   = A_abs[:, :, 1]
A_u10   = A_abs[:, :, 2]
A_v10   = A_abs[:, :, 3]
A_sst   = A_abs[:, :, 4]
A_sss   = A_abs[:, :, 5]
A_L2    = np.sqrt(np.sum(A2 ** 2, axis=2))

# ── ERA5 covariates on the RL grid ───────────────────────────────────────────
required_vars = ["t2m", "d2m", "cape", "blh", "u10", "v10"]
for v in required_vars:
    if v not in ds_instant:
        raise ValueError(f"Required ERA5 variable '{v}' not found.")

t2m_rl   = era_to_rl_nearest(to_2d(ds_instant["t2m"]))
d2m_rl   = era_to_rl_nearest(to_2d(ds_instant["d2m"]))
cape_rl  = era_to_rl_nearest(to_2d(ds_instant["cape"]))
blh_rl   = era_to_rl_nearest(to_2d(ds_instant["blh"]))
u10_rl   = era_to_rl_nearest(to_2d(ds_instant["u10"]))
v10_rl   = era_to_rl_nearest(to_2d(ds_instant["v10"]))

# Derived thermodynamic quantities
t_C   = t2m_rl - 273.15
td_C  = d2m_rl - 273.15
es    = np.exp((17.625 * t_C)  / (243.04 + t_C))
esd   = np.exp((17.625 * td_C) / (243.04 + td_C))
RH    = np.clip(100.0 * esd / np.maximum(es, 1e-12), 0.0, 100.0)
WSPD  = np.sqrt(u10_rl ** 2 + v10_rl ** 2)

# ── Continuous physical gates ────────────────────────────────────────────────

# Hygroscopic gate: warm, moist, and convectively active near-surface conditions
RH_ocean_vals = RH.copy()
RH_ocean_vals[~ocean_mask_rl] = np.nan
RH_threshold  = float(np.clip(np.nanquantile(RH_ocean_vals, 0.70), 55.0, 75.0))

gate_hygro = (
    sigmoid((t2m_rl  - 275.15)    /  2.0)
    * sigmoid((RH      - RH_threshold) /  5.0)
    * sigmoid((cape_rl - 5.0)         / 20.0)
)

# Dynamic gate: mechanically driven wind and boundary-layer depth
gate_dyn = (
    sigmoid((WSPD   - 4.0)   / 1.5)
    * sigmoid((blh_rl - 150.0) / 250.0)
)

# Glaciogenic gate: cold-band proxy from a simplified cloud-top temperature
z_eff_km   = np.clip((blh_rl / 1000.0) + 2.0, 1.0, 4.5)
T_cloud_C  = t_C - 6.5 * z_eff_km

WBAND            = 3.0
gate_below_p2    = sigmoid(( 2.0 - T_cloud_C) / WBAND)
gate_above_m20   = sigmoid((T_cloud_C + 20.0)  / WBAND)
cold_band        = gate_below_p2 * gate_above_m20
moist_gate       = sigmoid((RH      - 50.0) / 7.0)
cape_gate        = sigmoid((cape_rl - 15.0) / 25.0)
gate_glacio      = cold_band * moist_gate * cape_gate

temp_factor      = np.clip((2.0 - T_cloud_C) / 22.0, 0.0, 1.0)

# ── Mode score computation ───────────────────────────────────────────────────
A_hum_norm   = normalize_by_quantile(A_hum, q_hi=0.98)
wind_action  = normalize_by_quantile(np.sqrt(A_u10 ** 2 + A_v10 ** 2), q_hi=0.98)
RH_norm      = normalize_by_quantile(RH,      q_hi=0.95)
CAPE_norm    = normalize_by_quantile(cape_rl, q_hi=0.95)
BLH_norm     = normalize_by_quantile(blh_rl,  q_hi=0.95)
WSPD_norm    = normalize_by_quantile(WSPD,    q_hi=0.95)

W_HYGRO  = {"action": 0.55, "rh":   0.25, "cape": 0.20}
W_DYN    = {"action": 0.55, "wind": 0.25, "blh":  0.20}
W_GLACIO = {"action": 0.50, "cape": 0.25, "temp": 0.25}

score_hygro  = gate_hygro  * (W_HYGRO["action"]  * A_hum_norm
                               + W_HYGRO["rh"]    * RH_norm
                               + W_HYGRO["cape"]  * CAPE_norm)

score_dyn    = gate_dyn    * (W_DYN["action"]    * wind_action
                               + W_DYN["wind"]   * WSPD_norm
                               + W_DYN["blh"]    * BLH_norm)

score_glacio = gate_glacio * (W_GLACIO["action"] * A_hum_norm
                               + W_GLACIO["cape"] * CAPE_norm
                               + W_GLACIO["temp"] * temp_factor)

# Normalise to [0, 1] for visualization
H_plot = normalize_by_quantile(score_hygro,  0.98)
G_plot = normalize_by_quantile(score_glacio, 0.98)
D_plot = normalize_by_quantile(score_dyn,    0.98)
I_plot = normalize_by_quantile(A_L2,         0.98)

# Ocean-only fields for decision making
Hns = ocean_only(H_plot)
Gns = ocean_only(G_plot)
Dns = ocean_only(D_plot)
Ins = ocean_only(I_plot)

# ── Discrete technique assignment ────────────────────────────────────────────
TECH_NAMES = np.array(["HYGROSCOPIC", "GLACIOGENIC", "DYNAMIC"], dtype=object)
technique  = np.full((n_side, n_side), "NO-GO", dtype=object)

stack      = np.stack([Hns, Gns, Dns], axis=2)
stack_safe = np.where(np.isfinite(stack), stack, -1.0)
argmax     = np.argmax(stack_safe, axis=2)

for iy in range(n_side):
    for ix in range(n_side):
        if not ocean_mask_rl[iy, ix]:
            continue
        if (not np.isfinite(Ins[iy, ix])) or (Ins[iy, ix] < THR_INTENSITY):
            continue
        technique[iy, ix] = TECH_NAMES[argmax[iy, ix]]

# ── Score maps ───────────────────────────────────────────────────────────────
extent_rl = [float(lon_axis.min()), float(lon_axis.max()),
             float(lat_axis.min()), float(lat_axis.max())]

proj   = ccrs.PlateCarree()
fig, axs = plt.subplots(2, 2, figsize=(16, 10), subplot_kw={"projection": proj})
panels = [
    (H_plot, "Hygroscopic score"),
    (G_plot, "Glaciogenic score"),
    (D_plot, "Dynamic score"),
    (I_plot, "Action intensity ‖a‖₂"),
]
for ax, (fmap, title) in zip(axs.flat, panels):
    im = plot_smooth_field(ax, fmap, title, proj, extent_rl,
                           vmin=0.0, vmax=1.0, cmap=CMAP_SCORES)
    cb = plt.colorbar(im, ax=ax, orientation="vertical",
                      fraction=0.046, pad=0.2, location="left")
    cb.set_label("Relative intensity", fontsize=9)

plt.suptitle("Marine Cloud Seeding — Three-Mode Scores and Action Intensity", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
_path = os.path.join(FIGDIR, "stageD_techniques_heatmaps_smooth.png")
plt.savefig(_path, dpi=DPI_SAVE, bbox_inches="tight")
plt.show()
print(f"Saved : {_path}")

# ── Cell-level decision table ─────────────────────────────────────────────────
df_cells = pd.DataFrame({
    "lon":              LON2.flatten(),
    "lat":              LAT2.flatten(),
    "technique":        technique.flatten(),
    "score_hygro":      Hns.flatten(),
    "score_glacio":     Gns.flatten(),
    "score_dynamic":    Dns.flatten(),
    "action_intensity": Ins.flatten(),
    "t2m_K":            ocean_only(t2m_rl).flatten(),
    "RH2m_pct":         ocean_only(RH).flatten(),
    "cape":             ocean_only(cape_rl).flatten(),
    "blh":              ocean_only(blh_rl).flatten(),
    "wspd10":           ocean_only(WSPD).flatten(),
    "a_slp":            ocean_only(A_slp).flatten(),
    "a_humidity":       ocean_only(A_hum).flatten(),
    "a_u10":            ocean_only(A_u10).flatten(),
    "a_v10":            ocean_only(A_v10).flatten(),
    "a_sst":            ocean_only(A_sst).flatten(),
    "a_sss":            ocean_only(A_sss).flatten(),
})
df_cells.to_csv(os.path.join(CSVDIR, "seeding_cells_last_epoch.csv"), index=False)
print(f"Saved : {os.path.join(CSVDIR, 'seeding_cells_last_epoch.csv')}")

# ── Technique prevalence tables ───────────────────────────────────────────────
def technique_counts(df: pd.DataFrame, col: str = "technique") -> pd.DataFrame:
    counts = df[col].value_counts(dropna=False).reset_index()
    counts.columns = [col, "count"]
    counts["percent"] = 100.0 * counts["count"] / max(1, counts["count"].sum())
    return counts

ACTIVE_TECHS = ["HYGROSCOPIC", "GLACIOGENIC", "DYNAMIC"]

technique_counts(df_cells).to_csv(
    os.path.join(CSVDIR, "seeding_technique_counts_all_cells.csv"), index=False)

df_active = df_cells[df_cells["technique"].isin(ACTIVE_TECHS)].copy()
technique_counts(df_active).to_csv(
    os.path.join(CSVDIR, "seeding_technique_counts_active_cells.csv"), index=False)

# Per-technique descriptive statistics
SCORE_COLS = [
    "action_intensity", "score_hygro", "score_glacio", "score_dynamic",
    "wspd10", "RH2m_pct", "cape", "blh", "t2m_K",
]
stat_rows = []
for tech, grp in df_active.groupby("technique"):
    row = {"technique": tech, "count": int(len(grp))}
    for c in SCORE_COLS:
        v = pd.to_numeric(grp[c], errors="coerce")
        row[f"{c}_mean"]   = float(np.nanmean(v))
        row[f"{c}_median"] = float(np.nanmedian(v))
    stat_rows.append(row)

pd.DataFrame(stat_rows).sort_values("count", ascending=False).to_csv(
    os.path.join(CSVDIR, "seeding_stats_last_epoch.csv"), index=False)
print(f"Saved : {os.path.join(CSVDIR, 'seeding_stats_last_epoch.csv')}")

# Run-level report
run_report = {
    "device":      DEVICE,
    "base_in":     BASE_IN,
    "outdir":      OUTDIR,
    "era5_dir":    ERA5_DIR,
    "epochs_rl":   int(EPOCHS_RL),
    "thr_intensity": float(THR_INTENSITY),
    "ocean_fraction_era5_sst": float(ocean_ratio),
    "predictor_final_metrics": final_metrics,
    "notes": [
        "All figures saved at 400 DPI.",
        "All maps include longitude/latitude axis labels.",
        "Stage D uses continuous sigmoid gates to avoid technique collapse.",
        "Stage B uses the same train-scaled state representation as Stage A; "
        "state features are not scaled twice.",
        "Stage E always populates interpretability outputs; absent real ΔP, "
        "a conservative proxy is constructed from technique scores.",
    ],
}
save_json(os.path.join(OUTDIR, "run_report.json"), run_report)
print("[Stage D] Complete.")


# =============================================================================
# Stage E — Technique-Level Interpretability with ΔP Linkage
# =============================================================================
print("\n[Stage E] Interpretability — technique-level ΔP summaries")

TECHS = ["HYGROSCOPIC", "GLACIOGENIC", "DYNAMIC"]


# ── Helper functions ─────────────────────────────────────────────────────────

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: lowercase, strip whitespace, replace spaces."""
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_").lower() for c in df.columns]
    if df.columns.duplicated().any():
        new_cols, seen = [], {}
        for c in df.columns:
            if c not in seen:
                seen[c] = 0
                new_cols.append(c)
            else:
                seen[c] += 1
                new_cols.append(f"{c}__dup{seen[c]}")
        df.columns = new_cols
    return df


def detect_deltaP_column(df: pd.DataFrame):
    """
    Locate and normalize the ΔP column to mm/day.

    Accepted column names (in priority order):
      deltap_mmday, deltap_mmph (×24), deltap_mm_day, deltap_day,
      deltap, deltap_mm.

    Returns
    -------
    df_out : pd.DataFrame  with 'deltap_mmday' column added
    mode   : str           description of the source column
    """
    cols = df.columns
    if "deltap_mmday" in cols:
        df["deltap_mmday"] = pd.to_numeric(df["deltap_mmday"], errors="coerce")
        return df, "deltap_mmday"
    if "deltap_mmph" in cols:
        df["deltap_mmday"] = pd.to_numeric(df["deltap_mmph"], errors="coerce") * 24.0
        return df, "deltap_mmph_x24"
    for c in ("deltap_mm_day", "deltap_day", "deltap", "deltap_mm"):
        if c in cols:
            df["deltap_mmday"] = pd.to_numeric(df[c], errors="coerce")
            return df, f"{c}_assumed_mmday"
    raise ValueError("No ΔP column found. Expected deltap_mmday or deltap_mmph.")


def softmax_weights(scores: np.ndarray,
                    temperature: float = 0.15,
                    floor: float = 1e-3) -> np.ndarray:
    """
    Compute row-wise softmax weights with a temperature parameter and
    a uniform floor to ensure non-degenerate allocation.

    Parameters
    ----------
    scores      : (N, 3) array of technique scores per cell
    temperature : controls sharpness; lower = sharper
    floor       : minimum weight per technique (added before renormalisation)
    """
    x = np.array(scores, dtype=float)
    x[~np.isfinite(x)] = 0.0
    x = x / max(temperature, 1e-6)
    x = x - np.max(x, axis=1, keepdims=True)   # numerical stability
    ex = np.exp(x)
    w  = ex / (np.sum(ex, axis=1, keepdims=True) + 1e-12)
    w  = w + floor
    w  = w / (np.sum(w, axis=1, keepdims=True) + 1e-12)
    return w


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Compute a weighted quantile via sorted cumulative weight."""
    v = np.array(values, dtype=float)
    w = np.array(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not m.any():
        return np.nan
    v, w   = v[m], w[m]
    order  = np.argsort(v)
    v, w   = v[order], w[order]
    cw     = np.cumsum(w)
    return float(v[np.searchsorted(cw, q * cw[-1], side="left")])


def effective_n(weights: np.ndarray) -> float:
    """Kish effective sample size: (Σw)² / Σw²."""
    w = np.array(weights, dtype=float)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return 0.0
    return float(np.sum(w) ** 2 / (np.sum(w ** 2) + 1e-12))


def aggregate_weighted(dp: np.ndarray, w: np.ndarray) -> dict:
    """Weighted descriptive statistics for a ΔP vector."""
    dp = np.array(dp, dtype=float)
    w  = np.array(w,  dtype=float)
    m  = np.isfinite(dp) & np.isfinite(w) & (w > 0)
    if not m.any():
        return {k: np.nan for k in ("sum_w", "eff_n", "mean", "median",
                                    "q05", "q95", "pos_share")} | {"sum_w": 0.0, "eff_n": 0.0}
    dp, w    = dp[m], w[m]
    sw       = float(np.sum(w))
    return {
        "sum_w":     sw,
        "eff_n":     effective_n(w),
        "mean":      float(np.sum(w * dp) / (sw + 1e-12)),
        "median":    weighted_quantile(dp, w, 0.50),
        "q05":       weighted_quantile(dp, w, 0.05),
        "q95":       weighted_quantile(dp, w, 0.95),
        "pos_share": float(np.sum(w * (dp > 0.0)) / (sw + 1e-12)),
    }


# ── Locate or construct the ΔP-matched dataset ───────────────────────────────

def find_deltaP_candidate(root: str):
    """Search for a real ΔP linkage file under `root`, preferring matched-only files."""
    patterns = [
        os.path.join(root, "**", "*cells_with_deltaP*matched*only*.csv"),
        os.path.join(root, "**", "*deltap_cells_link*matched*only*.csv"),
        os.path.join(root, "**", "*deltap_cells_link*.csv"),
        os.path.join(root, "**", "*cells_with_deltaP*.csv"),
        os.path.join(root, "**", "*deltap*.csv"),
    ]
    hits = sorted({f for p in patterns for f in glob.glob(p, recursive=True)})
    hits = sorted(hits, key=lambda x: ("matched_only" not in x.lower(), len(x)))
    return hits[0] if hits else None


all_cells     = standardize_columns(pd.read_csv(
    os.path.join(CSVDIR, "seeding_cells_last_epoch.csv")))
real_candidate = find_deltaP_candidate(OUTDIR)

OUTPUT_MATCHED = os.path.join(INTERCSV, "cells_with_deltaP_matched_only.csv")

if real_candidate:
    print(f"[Stage E] Real ΔP file found: {real_candidate}")
    matched      = standardize_columns(pd.read_csv(real_candidate))
    if "lon" not in matched.columns or "lat" not in matched.columns:
        raise ValueError(f"ΔP file missing lon/lat columns: {real_candidate}")
    matched, delta_mode = detect_deltaP_column(matched)
    matched["delta_source"] = "real_file"
    delta_source = "real_file"
else:
    print("[Stage E] No real ΔP file found — constructing conservative proxy from scores.")
    needed = ["lon", "lat", "technique", "score_hygro",
              "score_glacio", "score_dynamic", "action_intensity"]
    for c in needed:
        if c not in all_cells.columns:
            raise ValueError(f"Stage D output missing required column '{c}'.")

    matched = all_cells.copy()
    for c in ["lon", "lat", "score_hygro", "score_glacio",
              "score_dynamic", "action_intensity"]:
        matched[c] = pd.to_numeric(matched[c], errors="coerce")

    matched = matched[matched["technique"].isin(TECHS)].copy()
    matched = matched[np.isfinite(matched["action_intensity"])
                      & (matched["action_intensity"] >= THR_INTENSITY)].copy()
    matched = matched[np.isfinite(matched["lon"]) & np.isfinite(matched["lat"])].copy()

    score_max = np.nanmax(np.stack([
        matched["score_hygro"].to_numpy(dtype=float),
        matched["score_glacio"].to_numpy(dtype=float),
        matched["score_dynamic"].to_numpy(dtype=float),
    ], axis=1), axis=1)
    score_max  = np.clip(score_max, 0.0, 1.0)
    intensity  = np.clip(matched["action_intensity"].to_numpy(dtype=float), 0.0, 1.0)

    PROXY_MAX_MM_DAY = 4.0
    matched["deltap_mmday"] = PROXY_MAX_MM_DAY * (0.65 * score_max + 0.35 * intensity)
    matched["delta_source"] = "proxy_from_scores"
    delta_mode   = "proxy_scores_to_mmday"
    delta_source = "proxy_from_scores"

matched.to_csv(OUTPUT_MATCHED, index=False)
print(f"Saved : {OUTPUT_MATCHED}")

# ── Attach scores to matched dataset if missing ──────────────────────────────
SCORE_COLS_E = ["score_hygro", "score_glacio", "score_dynamic", "action_intensity"]
scores_in_matched  = all(c in matched.columns  for c in SCORE_COLS_E)
scores_in_allcells = all(c in all_cells.columns for c in SCORE_COLS_E)

if not scores_in_matched:
    if not scores_in_allcells:
        raise ValueError("Scores unavailable in both matched dataset and all-cells table.")

    for c in SCORE_COLS_E:
        all_cells[c] = pd.to_numeric(all_cells[c], errors="coerce")
    for df in (all_cells, matched):
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")

    # Nearest-neighbor join
    try:
        from sklearn.neighbors import BallTree
        A      = np.deg2rad(all_cells[["lat", "lon"]].to_numpy())
        B      = np.deg2rad(matched[["lat", "lon"]].to_numpy())
        tree   = BallTree(A, metric="haversine")
        dist, idx_nn = tree.query(B, k=1)
        dist_km = dist[:, 0] * 6371.0
        idx_nn  = idx_nn[:, 0]
    except Exception:
        A = all_cells[["lon", "lat"]].to_numpy()
        B = matched[["lon", "lat"]].to_numpy()
        idx_nn, dist_km = [], []
        for row in B:
            d2 = np.sum((A - row) ** 2, axis=1)
            j  = int(np.argmin(d2))
            idx_nn.append(j)
            dist_km.append(float(np.sqrt(d2[j])) * 111.0)
        idx_nn  = np.array(idx_nn, dtype=int)
        dist_km = np.array(dist_km, dtype=float)

    matched["nn_dist_km"]   = dist_km
    matched["nn_bad_match"] = (
        (~np.isfinite(dist_km)) | (dist_km > 60.0)
    ).astype(int)
    ref = all_cells.iloc[idx_nn].reset_index(drop=True)
    for c in SCORE_COLS_E:
        matched[c] = ref[c].to_numpy()
else:
    matched["nn_dist_km"]   = np.nan
    matched["nn_bad_match"] = 0

# Re-detect ΔP column after potential updates
matched = standardize_columns(matched)
matched, delta_mode_final = detect_deltaP_column(matched)

# Intensity gate for the soft-attribution subset
matched["gate_intensity"] = (
    np.isfinite(matched["action_intensity"])
    & (matched["action_intensity"] >= THR_INTENSITY)
).astype(int)
matched_gate1 = matched[matched["gate_intensity"] == 1].copy()

# ── Soft-weighted ΔP summaries ────────────────────────────────────────────────
score_matrix = np.stack([
    pd.to_numeric(matched_gate1["score_hygro"],  errors="coerce").to_numpy(dtype=float),
    pd.to_numeric(matched_gate1["score_glacio"], errors="coerce").to_numpy(dtype=float),
    pd.to_numeric(matched_gate1["score_dynamic"],errors="coerce").to_numpy(dtype=float),
], axis=1)

W = softmax_weights(score_matrix, temperature=0.15, floor=1e-3)
matched_gate1["w_hygro"]   = W[:, 0]
matched_gate1["w_glacio"]  = W[:, 1]
matched_gate1["w_dynamic"] = W[:, 2]

dp_gate1 = pd.to_numeric(matched_gate1["deltap_mmday"], errors="coerce").to_numpy(dtype=float)

soft_rows = []
for tech, wcol in [("HYGROSCOPIC", "w_hygro"),
                   ("GLACIOGENIC", "w_glacio"),
                   ("DYNAMIC",     "w_dynamic")]:
    w = pd.to_numeric(matched_gate1[wcol], errors="coerce").to_numpy(dtype=float)
    st = aggregate_weighted(dp_gate1, w)
    st["technique"] = tech
    soft_rows.append(st)

df_soft = pd.DataFrame(soft_rows)[
    ["technique", "sum_w", "eff_n", "mean", "median", "q05", "q95", "pos_share"]
]

# Weight diagnostics
_n   = len(matched_gate1)
diag = pd.DataFrame({
    "technique": TECHS,
    "w_min": [float(np.nanmin(matched_gate1[c])) if _n else np.nan
              for c in ("w_hygro", "w_glacio", "w_dynamic")],
    "w_med": [float(np.nanmedian(matched_gate1[c])) if _n else np.nan
              for c in ("w_hygro", "w_glacio", "w_dynamic")],
    "w_max": [float(np.nanmax(matched_gate1[c])) if _n else np.nan
              for c in ("w_hygro", "w_glacio", "w_dynamic")],
    "sum_w_gate1": [float(np.nansum(matched_gate1[c])) if _n else 0.0
                    for c in ("w_hygro", "w_glacio", "w_dynamic")],
    "n_gate1_rows": int(_n),
})

# Hard attribution counts (for comparison)
if "technique" in matched_gate1.columns:
    vc  = matched_gate1["technique"].value_counts(dropna=False)
    tot = float(vc.sum()) or 1.0
    hard_tbl = pd.DataFrame([{
        "technique": t,
        "count":     int(vc.get(t, 0)),
        "percent":   100.0 * float(vc.get(t, 0)) / tot,
    } for t in TECHS])
else:
    hard_tbl = pd.DataFrame([{"technique": t, "count": 0, "percent": 0.0}
                              for t in TECHS])

# ── Save Stage E outputs ─────────────────────────────────────────────────────
outputs_e = {
    "deltaP_by_technique_matched_only.csv":          df_soft,
    "deltaP_by_technique_soft.csv":                  df_soft,
    "deltaP_by_technique_hard_matched_only.csv":     hard_tbl,
    "weights_diagnostics_soft.csv":                  diag,
    "cells_with_deltaP_matched_only_enriched_soft.csv": matched_gate1,
}
for filename, df_out in outputs_e.items():
    path = os.path.join(INTERCSV, filename)
    df_out.to_csv(path, index=False)
    print(f"Saved : {path}")

report_e = {
    "stage":         "E",
    "outdir":        OUTDIR,
    "delta_source":  delta_source,
    "delta_mode":    delta_mode if delta_source == "proxy_from_scores" else delta_mode_final,
    "delta_units":   "mm/day",
    "rows_matched_total": int(len(matched)),
    "rows_gate1_used":    int(_n),
    "thr_intensity":      float(THR_INTENSITY),
    "inputs": {
        "all_cells_stage_d":  os.path.join(CSVDIR, "seeding_cells_last_epoch.csv"),
        "real_candidate":     real_candidate,
        "matched_written":    OUTPUT_MATCHED,
    },
    "outputs": {k: os.path.join(INTERCSV, k) for k in outputs_e},
    "note": (
        "When no real ΔP linkage file is found, a conservative proxy is derived "
        "from technique scores and action intensity. It is flagged as "
        "'delta_source: proxy_from_scores' and should not be used for causal attribution."
    ),
}
save_json(os.path.join(INTERCSV, "interpretability_pack_report_soft.json"), report_e)

# ── Summary printout ──────────────────────────────────────────────────────────
print("\n[Stage E] Soft ΔP by technique (mm/day):")
print(df_soft.to_string(index=False))
print("\n[Stage E] Weight diagnostics:")
print(diag.to_string(index=False))
print("\n[Stage E] Hard attribution (comparison):")
print(hard_tbl.to_string(index=False))
print(f"\n[Stage E] Complete. Outputs in: {INTERCSV}")

print("\n" + "=" * 70)
print("Pipeline complete — all stages (0, A, B, C, D, E) finished.")
print("=" * 70)