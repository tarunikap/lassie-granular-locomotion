import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

raw_file = r"D:\Awesomnesss\Summer '25 Intern\USC\LASSIE\LASSIE_DataForHeatMap\LASSIE_DataForHeatMap\C toe\leg_d3\Cleg_d3_t4.csv" # your raw CSV (with ODrive 2 Position, ODrive 2 Current)
seg_file = r"D:\Awesomnesss\Summer '25 Intern\USC\LASSIE\toe_segments_output.csv" # segmentation CSV (with motor_pos, x_mid, z_mid, beta_deg, gamma_deg, depth_m, in_ground)
alpha_outfile = r"D:\Awesomnesss\Summer '25 Intern\USC\LASSIE\alpha_bins.csv"

# physical params
K_t = 0.0973       # torque constant (Nm/A)
R_eff = 0.05       # effective lever arm used only to convert torque->force scale (m); used only if you want unit scaling
thickness = 0.01   # thickness for element area (m) -> dA = arc_len * thickness
R = 0.05           # toe radius used to compute arc length if needed (m)
Nseg_default = 100  # segmentation N used when dA = R * (pi/N) * thickness

# binning for beta,gamma (user requested -180 .. 80)
bin_size = 5  # degrees (change to 5 or 15 if you want finer/coarser)
beta_min, beta_max = -180.0, 180.0
gamma_min, gamma_max = -180.0, 180.0

# regularization
lambda_rel = 1e-6  # ridge regularization scale (increase if solution noisy)

# ============================
# Read input data
# ============================
raw_df = pd.read_csv(raw_file)
raw_df.columns = [c.strip() for c in raw_df.columns]

seg_df = pd.read_csv(seg_file)
seg_df.columns = [c.strip() for c in seg_df.columns]

# Check necessary columns
if "ODrive 2 Position" not in raw_df.columns or "ODrive 2 Current" not in raw_df.columns:
    raise ValueError("raw CSV must contain columns: 'ODrive 2 Position' and 'ODrive 2 Current'")

# segmentation expected columns: motor_pos, x_mid, z_mid, beta_deg, gamma_deg, depth_m, in_ground
if "motor_pos" not in seg_df.columns:
    # try some alternatives
    possible = [c for c in seg_df.columns if "motor" in c.lower() and "pos" in c.lower()]
    if possible:
        seg_df = seg_df.rename(columns={possible[0]: "motor_pos"})
    else:
        raise ValueError("segmentation CSV must contain column 'motor_pos' (or similar)")

required_seg_cols = ["x_mid", "z_mid", "beta_deg", "gamma_deg", "depth_m", "in_ground"]
for c in required_seg_cols:
    if c not in seg_df.columns:
        raise ValueError(f"segmentation CSV missing required column: {c}")

# ============================
# Preprocessing / match currents
# ============================
# Sort both for merge-asof style matching
raw_sorted = raw_df.sort_values("ODrive 2 Position").reset_index(drop=True)
seg_sorted = seg_df.sort_values("motor_pos").reset_index(drop=True)

# estimate segmentation step to set tolerance
unique_positions = np.sort(seg_sorted["motor_pos"].unique())
if len(unique_positions) > 1:
    diffs = np.diff(unique_positions)
    seg_step = float(np.nanmin(diffs[diffs > 0])) if np.any(diffs > 0) else 1e-6
else:
    seg_step = 1e-3
tolerance = max(seg_step * 0.6, 1e-6)  # allow nearest within about half step

# We'll attach nearest current from raw to each seg row (nearest pos within tolerance)
# Using a simple vectorized nearest lookup (safe for arbitrary ordering)
raw_pos = raw_sorted["ODrive 2 Position"].values
raw_curr = raw_sorted["ODrive 2 Current"].values

# for speed, build an index by searchsorted:
idxs = np.searchsorted(raw_pos, seg_sorted["motor_pos"].values)
# clip idxs into valid range and check nearest
curr_for_seg = []
for s_pos, idx in zip(seg_sorted["motor_pos"].values, idxs):
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(raw_pos):
        candidates.append(idx)
    # choose best candidate by distance
    best = None
    best_dist = 1e9
    for c in candidates:
        d = abs(raw_pos[c] - s_pos)
        if d < best_dist:
            best = c
            best_dist = d
    if best is None or best_dist > max(tolerance, 0.05):  # fallback threshold 0.05 if something weird
        curr_for_seg.append(np.nan)
    else:
        curr_for_seg.append(raw_curr[best])

seg_sorted = seg_sorted.copy()
seg_sorted["ODrive_2_Current_matched"] = curr_for_seg

n_missing = seg_sorted["ODrive_2_Current_matched"].isna().sum()
if n_missing:
    print(f"Warning: {n_missing} segmentation rows had no nearby raw sample (they will be dropped).")

# drop rows without matched current
seg_matched = seg_sorted.dropna(subset=["ODrive_2_Current_matched"]).reset_index(drop=True)

# ============================
# Build beta/gamma bins
# ============================
beta_edges = np.arange(beta_min, beta_max + bin_size, bin_size)
gamma_edges = np.arange(gamma_min, gamma_max + bin_size, bin_size)
beta_centers = 0.5 * (beta_edges[:-1] + beta_edges[1:])
gamma_centers = 0.5 * (gamma_edges[:-1] + gamma_edges[1:])
n_b = len(beta_centers)
n_g = len(gamma_centers)
K = n_b * n_g  # bins count

print(f"Using bin size {bin_size}° → {n_b} x {n_g} = {K} (β x γ) bins")

# helper to map (b_idx,g_idx) -> flat index
def bin_index(bi, gi):
    return int(bi * n_g + gi)

# compute segment dA (approx): arc_len * thickness
# if seg file contains explicit arc length (ds) we can use it; otherwise estimate by R and segmentation count
if "ds" in seg_matched.columns:
    seg_matched["dA"] = seg_matched["ds"].astype(float) * thickness
else:
    # attempt to infer N from repeating counts per motor_pos (if segmentation created by you)
    # but safe fallback: use Nseg_default
    dtheta = math.pi / Nseg_default
    arc_len = R * dtheta
    seg_matched["dA"] = arc_len * thickness

# ============================
# Build linear system X alpha = b
# Unknowns: alpha_x for each (b,g) and alpha_z for each (b,g)
# ordering: [alpha_x_bin0..alpha_x_binK-1, alpha_z_bin0..alpha_z_binK-1]
# Row per unique motor_pos (pose): sum over segments in that pose contributes to coefficients
# coeff for alpha_x(k) at pose p = sum_{i in pose and bin k} (- z_i * dA_i * depth_i)
# coeff for alpha_z(k) at pose p = sum_{i in pose and bin k} (  x_i * dA_i * depth_i)
# ============================
poses = seg_matched["motor_pos"].unique()
poses.sort()
n_poses = len(poses)
print(f"Number of poses (unique motor_pos in seg file): {n_poses}")

X_rows = []
b_rows = []

# iterate poses
for p in poses:
    segs = seg_matched[seg_matched["motor_pos"] == p]
    # use the matched current from any seg (they are same by merge logic); use mean to be safe
    current_mean = segs["ODrive_2_Current_matched"].astype(float).mean()
    tau_meas = float(current_mean) * K_t  # measured motor torque [Nm]

    # initialize coefficient arrays for alpha_x and alpha_z bins
    coeff_x = np.zeros(K, dtype=float)
    coeff_z = np.zeros(K, dtype=float)

    # iterate segments within pose
    for _, s in segs.iterrows():
        if int(s["in_ground"]) == 0 or float(s["depth_m"]) <= 0:
            continue
        b = float(s["beta_deg"])
        g = float(s["gamma_deg"])
        # find bin indices
        bi = np.digitize([b], beta_edges)[0] - 1
        gi = np.digitize([g], gamma_edges)[0] - 1
        if not (0 <= bi < n_b and 0 <= gi < n_g):
            # skip segments outside range
            continue
        k_idx = bin_index(bi, gi)

        x_i = float(s["x_mid"])
        z_i = float(s["z_mid"])
        dA_i = float(s["dA"])
        depth_i = float(s["depth_m"])

        add = dA_i * depth_i
        coeff_x[k_idx] += - z_i * add   # multiplies alpha_x
        coeff_z[k_idx] +=   x_i * add   # multiplies alpha_z

    # combine row
    row = np.concatenate([coeff_x, coeff_z])
    X_rows.append(row)
    b_rows.append(tau_meas)

X = np.vstack(X_rows)         # shape (n_poses, 2K)
b = np.array(b_rows)          # shape (n_poses,)

print(f"Built system X shape = {X.shape}, b length = {b.shape[0]}")

# quick check: if X is all zeros something wrong
if np.allclose(X, 0):
    raise RuntimeError("All coefficients in X are zero — check segmentation (depth/in_ground) and bin ranges.")

# ============================
# Solve regularized least squares: minimize ||X alpha - b||^2 + lambda ||alpha||^2
# Solve (X^T X + lambda I) alpha = X^T b
# ============================
XtX = X.T @ X
Xtb = X.T @ b

# set lambda relative scale
diag_scale = np.max(np.diag(XtX)) if np.max(np.diag(XtX)) > 0 else 1.0
lam = lambda_rel * diag_scale
A_reg = XtX + lam * np.eye(XtX.shape[0])

# try solve; if singular fallback to lstsq
try:
    alpha_all = np.linalg.solve(A_reg, Xtb)
except np.linalg.LinAlgError:
    print("Matrix singular in direct solve, using np.linalg.lstsq fallback")
    alpha_all, *_ = np.linalg.lstsq(X, b, rcond=None)

# predicted and residuals
b_pred = X @ alpha_all
residuals = b - b_pred
rmse = np.sqrt(np.mean(residuals**2))
print(f"RMSE on torque (Nm): {rmse:.6e}")

# ============================
# Extract alpha_x and alpha_z into 2D arrays and save CSV
# ============================
alpha_x = alpha_all[:K].reshape((n_b, n_g))
alpha_z = alpha_all[K: ].reshape((n_b, n_g))

# build DataFrame of bins
rows_out = []
for bi in range(n_b):
    for gi in range(n_g):
        rows_out.append({
            "beta_center_deg": beta_centers[bi],
            "gamma_center_deg": gamma_centers[gi],
            "alpha_x": alpha_x[bi, gi],
            "alpha_z": alpha_z[bi, gi],
        })
df_bins = pd.DataFrame(rows_out)
df_bins.to_csv(alpha_outfile, index=False)
print(f"Saved alpha per-bin CSV: {alpha_outfile}")

# ============================
# Plot results (like Chen Li)
# ============================
Bc, Gc = np.meshgrid(gamma_centers, beta_centers)  # note: meshgrid order matches alpha_x/alpha_z shapes

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
# alpha_z map
pcm = plt.pcolormesh(Gc, Bc, alpha_z, shading='auto', cmap='RdYlBu')
plt.colorbar(pcm, label='alpha_z (units)')
plt.xlabel('gamma (deg)')
plt.ylabel('beta (deg)')
plt.title('alpha_z (beta,gamma)')

plt.subplot(1,2,2)
pcm2 = plt.pcolormesh(Gc, Bc, alpha_x, shading='auto', cmap='RdYlGn')
plt.colorbar(pcm2, label='alpha_x (units)')
plt.xlabel('gamma (deg)')
plt.ylabel('beta (deg)')
plt.title('alpha_x (beta,gamma)')
plt.tight_layout()
plt.show()

# Print a brief summary
print("Done. Keys:")
print(f"  - bins: beta {beta_min}..{beta_max} step {bin_size}  |  gamma {gamma_min}..{gamma_max} step {bin_size}")
print(f"  - solved alpha for {K} bins (x and z), wrote: {alpha_outfile}")
