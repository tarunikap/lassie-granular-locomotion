import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

raw_file = r"D:\Awesomnesss\Summer '25 Intern\USC\LASSIE\LASSIE_DataForHeatMap\LASSIE_DataForHeatMap\C toe\leg_d3\Cleg_d3_t4.csv"
seg_file = r"D:\Awesomnesss\Summer '25 Intern\USC\LASSIE\toe_segments_output.csv"
alpha_outfile = r"D:\Awesomnesss\Summer '25 Intern\USC\LASSIE\alpha_bins.csv"

K_t = 0.0973       # torque constant (Nm/A)
R_eff = 0.05       # effective lever arm
thickness = 0.01   # thickness for element area (m)
R = 0.05           # toe radius
Nseg_default = 100 # segmentation count

# binning for beta,gamma
bin_size = 5
beta_min, beta_max = -180.0, 180.0
gamma_min, gamma_max = -180.0, 180.0

lambda_rel = 1e-6

raw_df = pd.read_csv(raw_file)
raw_df.columns = [c.strip() for c in raw_df.columns]

seg_df = pd.read_csv(seg_file)
seg_df.columns = [c.strip() for c in seg_df.columns]

if "ODrive 2 Position" not in raw_df.columns or "ODrive 2 Current" not in raw_df.columns:
    raise ValueError("raw CSV must contain 'ODrive 2 Position' and 'ODrive 2 Current'")

if "motor_pos" not in seg_df.columns:
    possible = [c for c in seg_df.columns if "motor" in c.lower() and "pos" in c.lower()]
    if possible:
        seg_df = seg_df.rename(columns={possible[0]: "motor_pos"})
    else:
        raise ValueError("segmentation CSV must contain 'motor_pos'")

required_seg_cols = ["x_mid", "z_mid", "beta_deg", "gamma_deg", "depth_m", "in_ground"]
for c in required_seg_cols:
    if c not in seg_df.columns:
        raise ValueError(f"segmentation CSV missing {c}")


raw_sorted = raw_df.sort_values("ODrive 2 Position").reset_index(drop=True)
seg_sorted = seg_df.sort_values("motor_pos").reset_index(drop=True)

raw_pos = raw_sorted["ODrive 2 Position"].values
raw_curr = raw_sorted["ODrive 2 Current"].values
seg_pos = seg_sorted["motor_pos"].values

curr_for_seg = []
for s_pos in seg_pos:
    idx = np.searchsorted(raw_pos, s_pos)
    if idx == 0:
        curr_val = raw_curr[0]
    elif idx >= len(raw_pos):
        curr_val = raw_curr[-1]
    else:
        x0, x1 = raw_pos[idx-1], raw_pos[idx]
        y0, y1 = raw_curr[idx-1], raw_curr[idx]
        curr_val = y0 if x1==x0 else y0 + (s_pos - x0)/(x1 - x0) * (y1 - y0)
    curr_for_seg.append(curr_val)

seg_sorted["ODrive_2_Current_matched"] = curr_for_seg


beta_edges = np.arange(beta_min, beta_max + bin_size, bin_size)
gamma_edges = np.arange(gamma_min, gamma_max + bin_size, bin_size)
beta_centers = 0.5 * (beta_edges[:-1] + beta_edges[1:])
gamma_centers = 0.5 * (gamma_edges[:-1] + gamma_edges[1:])
n_b = len(beta_centers)
n_g = len(gamma_centers)
K = n_b * n_g

def bin_index(bi, gi):
    return int(bi * n_g + gi)

# compute segment dA
if "ds" in seg_sorted.columns:
    seg_sorted["dA"] = seg_sorted["ds"].astype(float) * thickness
else:
    dtheta = math.pi / Nseg_default
    arc_len = R * dtheta
    seg_sorted["dA"] = arc_len * thickness


used_positions = set()
X_rows, b_rows = [], []

raw_pos_rounded_unique = np.round(raw_sorted["ODrive 2 Position"].values, 2)
for p_round in raw_pos_rounded_unique:
    if p_round in used_positions:
        continue
    used_positions.add(p_round)
    
    segs = seg_sorted[np.round(seg_sorted["motor_pos"].values, 2) == p_round]
    segs = segs.dropna(subset=["ODrive_2_Current_matched"])
    if len(segs) == 0:
        continue
    
    tau_meas = float(segs["ODrive_2_Current_matched"].mean()) * K_t
    
    coeff_x = np.zeros(K, dtype=float)
    coeff_z = np.zeros(K, dtype=float)
    
    for _, s in segs.iterrows():
        if int(s["in_ground"]) == 0 or float(s["depth_m"]) <= 0:
            continue
        b = float(s["beta_deg"])
        g = float(s["gamma_deg"])
        bi = np.digitize([b], beta_edges)[0] - 1
        gi = np.digitize([g], gamma_edges)[0] - 1
        if not (0 <= bi < n_b and 0 <= gi < n_g):
            continue
        k_idx = bin_index(bi, gi)
        
        x_i = float(s["x_mid"])
        z_i = float(s["z_mid"])
        dA_i = float(s["dA"])
        depth_i = float(s["depth_m"])
        add = dA_i * depth_i
        
        coeff_x[k_idx] += - z_i * add
        coeff_z[k_idx] +=   x_i * add
    
    row = np.concatenate([coeff_x, coeff_z])
    X_rows.append(row)
    b_rows.append(tau_meas)

# Solve linear system
X = np.vstack(X_rows)
b = np.array(b_rows)

XtX = X.T @ X
Xtb = X.T @ b
diag_scale = np.max(np.diag(XtX)) if np.max(np.diag(XtX)) > 0 else 1.0
lam = lambda_rel * diag_scale
A_reg = XtX + lam * np.eye(XtX.shape[0])

try:
    alpha_all = np.linalg.solve(A_reg, Xtb)
except np.linalg.LinAlgError:
    alpha_all, *_ = np.linalg.lstsq(X, b, rcond=None)

b_pred = X @ alpha_all
residuals = b - b_pred
rmse = np.sqrt(np.mean(residuals**2))
print(f"RMSE on torque (Nm): {rmse:.6e}")


alpha_x = alpha_all[:K].reshape((n_b, n_g))
alpha_z = alpha_all[K:].reshape((n_b, n_g))

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


Bc, Gc = np.meshgrid(gamma_centers, beta_centers)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
pcm = plt.pcolormesh(Gc, Bc, alpha_z, shading='auto', cmap='RdYlBu')
plt.colorbar(pcm, label='alpha_z')
plt.xlabel('gamma (deg)')
plt.ylabel('beta (deg)')
plt.title('alpha_z (beta,gamma)')

plt.subplot(1,2,2)
pcm2 = plt.pcolormesh(Gc, Bc, alpha_x, shading='auto', cmap='RdYlGn')
plt.colorbar(pcm2, label='alpha_x')
plt.xlabel('gamma (deg)')
plt.ylabel('beta (deg)')
plt.title('alpha_x (beta,gamma)')
plt.tight_layout()
plt.show()

print("All segmentation rows used without repeating rounded motor positions.")
