import numpy as np

beta_deg = np.array([-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90])
gamma_deg = np.array([-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90])

beta = np.radians(beta_deg)   # shape (13,)
gamma = np.radians(gamma_deg) # shape (13,)

ax = np.array([
    [-0.09093, 0.1671, 0.0873, 0.0098, -0.2994, 0, 0, 0, 1.2941, 1.0146, 0.9838, 0.9619, -0.1478],
    [-0.3417, 0.0475, 0.0078, 0.3167, -0.1473, 0, 0, 0, 0.7733, 1.6392, 1.1441, 1.2528, 0.4356],
    [-0.9111, -0.4965, -0.1151, 0.0064, 0.1262, 0, 0, 0, 0.907, 1.4171, 1.5348, 1.2276, 0.6032],
    [0.0788, -0.8872, -0.5739, -0.1268, -0.0853, 0, 0, 0, 1.0344, 1.3693, 1.5521, 1.1666, 0.5937],
    [0.0799, 0.0993, -0.7632, -0.695, 0.2864, 0, 0, 0, 0.9539, 1.2875, 1.0272, 1.3105, 0.4382],
    [0.2649, 0.0773, -0.2217, -0.7118, 0.5112, 0, 0, 0, 0.7764, 1.094, 1.3137, 0.3909, 0.4921],
    [-0.0081, -0.2029, -0.0039, -0.6672, 0.4715, 0, 0, 0, 0.8285, 1.01, 0.7879, 0.5705, -0.2422],
    [-0.2649, -0.0156, -0.3924, -0.0882, -0.0793, 0, 0, 0, 0.5046, 0.8272, 0.5789, 0.2742, -0.4921],
    [-0.0799, -0.624, -0.0438, -0.4189, -0.0657, 0, 0, 0, 0.3928, 0.5283, 0.3637, 0.0552, -0.4382],
    [-0.0788, -0.1488, -0.8144, -0.113, -0.2691, 0, 0, 0, 0.5415, 0.5939, 0.2794, -0.0061, -0.5937],
    [0.9111, 0.0915, -0.2414, -0.7269, -0.0566, 0, 0, 0, 0.8222, 0.8403, 0.524, -0.0787, -0.6032],
    [0.3417, 0.4168, 0.0255, -0.2426, -0.678, 0, 0, 0, 1.103, 1.0606, 0.9039, 0.1202, -0.4356],
    [-0.09093, 0.1671, 0.0873, 0.0098, -0.2994, 0, 0, 0, 1.2941, 1.0146, 0.9838, 0.9619, -0.1478]
]) / 12.0

ay = np.array([
    [-0.1346, -0.1136, -0.0849, -0.1006, 0.0026, 0, 0, 0, -0.51, -0.5789, -0.9666, -1.8677, -1.3084],
    [-0.052, -0.1103, -0.1386, -0.2084, -0.0804, 0, 0, 0, -0.1966, -1.1841, -1.24, -2.3293, -2.0815],
    [0.1468, 0.0234, -0.3911, -0.1235, -0.2004, 0, 0, 0, -0.794, -1.203, -1.9337, -2.4324, -2.8374],
    [-0.0297, 0.3858, 0.1536, -0.0682, -0.1527, 0, 0, 0, -1.0481, -1.7964, -2.1793, -2.8013, -3.0851],
    [-0.0787, -0.052, 0.6183, 0.4667, -0.0259, 0, 0, 0, -1.1648, -1.8582, -2.4113, -3.275, -3.1554],
    [-0.1675, -0.0583, 0.0733, 0.736, 0.5384, 0, 0, 0, -1.1937, -1.8645, -3.3166, -2.784, -3.8658],
    [-0.0025, -0.1456, -0.1622, 0.3217, 0.5309, 0, 0, 0, -1.2083, -1.7793, -2.3312, -4.7097, -3.0321],
    [-0.1675, -0.0857, -0.2079, -0.1156, -0.0714, 0, 0, 0, -0.9384, -1.7791, -2.1776, -2.921, -3.8658],
    [-0.0787, -0.1906, -0.1296, -0.1726, -0.0384, 0, 0, 0, -0.3731, -1.2544, -1.9952, -2.5172, -3.1554],
    [-0.0297, -0.1128, -0.1908, -0.1376, -0.0237, 0, 0, 0, -0.1008, -0.8457, -1.3566, -2.076, -3.0851],
    [0.1468, -0.0929, -0.1688, -0.0922, -0.024, 0, 0, 0, -0.1047, -0.474, -1.2913, -1.4279, -2.8374],
    [-0.0525, -0.0363, -0.0886, -0.1208, 0.1095, 0, 0, 0, -0.322, -0.5291, -0.9606, -1.0411, -2.0815],
    [-0.1346, -0.1136, -0.0849, -0.1006, 0.0026, 0, 0, 0, -0.51, -0.5789, -0.9666, -1.8677, -1.3084]
]) / -12.0

ax[ax == 0.0] = np.nan
ay[ay == 0.0] = np.nan

def fit_truncated_fourier(data):
  
    B, G = np.meshgrid(beta, gamma, indexing='ij') 
    mask = ~np.isnan(data)
    y = data[mask].flatten()

    m_vals = [-1, 0, 1]
    n_vals = [0, 1]
    basis_list = []
    labels = []

    for m in m_vals:
        for n in n_vals:
            arg = 2*np.pi*( (m * B / np.pi) + (n * G / (2*np.pi)) )  # dimensionless
            c = np.cos(arg)[mask]
            s = np.sin(arg)[mask]
            basis_list.append(c); labels.append(f"COS_m{m}_n{n}")
            basis_list.append(s); labels.append(f"SIN_m{m}_n{n}")

    X = np.vstack(basis_list).T    # shape (N_samples, 18)
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)  # least-squares

    # Map coefficients to paper labels:
    mapping = {}
    # For alpha_z (A,B in paper) we will return labels A_mn / B_mn ; order depends on loop
    # We'll extract numeric values in a dictionary keyed by (m,n,cos/sin)
    idx = 0
    for m in m_vals:
        for n in n_vals:
            mapping[(m,n,'cos')] = coeffs[idx]; idx += 1
            mapping[(m,n,'sin')] = coeffs[idx]; idx += 1
    return mapping

# Fit for alpha_z (A,B) using ay, and for alpha_x (C,D) using ax
map_ab = fit_truncated_fourier(ay)  # A_mn (cos) and B_mn (sin)
map_cd = fit_truncated_fourier(ax)  # C_mn (cos) and D_mn (sin)

# According to table they list: A0,0  A1,0  B1,1  B0,1  B-1,1  C1,1  C0,1  C-1,1  D1,0
result = {}

# A0,0  -> (m=0,n=0, cos)
result['A0,0'] = map_ab[(0,0,'cos')]
# A1,0  -> (m=1,n=0, cos)
result['A1,0'] = map_ab[(1,0,'cos')]
# B1,1  -> (m=1,n=1, sin)
result['B1,1'] = map_ab[(1,1,'sin')]
# B0,1  -> (m=0,n=1, sin)
result['B0,1'] = map_ab[(0,1,'sin')]
# B-1,1 -> (m=-1,n=1, sin)
result['B-1,1'] = map_ab[(-1,1,'sin')]

# C1,1 -> alpha_x cos term (m=1,n=1, cos)
result['C1,1'] = map_cd[(1,1,'cos')]
# C0,1 -> (m=0,n=1, cos)
result['C0,1'] = map_cd[(0,1,'cos')]
# C-1,1 -> (m=-1,n=1, cos)
result['C-1,1'] = map_cd[(-1,1,'cos')]
# D1,0 -> (m=1,n=0, sin)
result['D1,0'] = map_cd[(1,0,'sin')]


try:
    b_idx = int(np.where(beta_deg == 0)[0][0])
    g_idx = int(np.where(gamma_deg == 90)[0][0])
except Exception:
    b_idx = None; g_idx = None

def evaluate_alpha_z_at(b_val, g_val, mapping):
    s = 0.0
    for m in [-1,0,1]:
        for n in [0,1]:
            arg = 2*np.pi*( (m * b_val / np.pi) + (n * g_val / (2*np.pi)) )
            A = mapping[(m,n,'cos')]
            B = mapping[(m,n,'sin')]
            s += A * np.cos(arg) + B * np.sin(arg)
    return s

if (b_idx is None) or (g_idx is None):
    print("Warning: can't find beta=0 or gamma=90 index in angle arrays. zeta not computed.")
    result['zeta'] = np.nan
else:
    alpha_z_measured = ay[b_idx, g_idx]   # may be NaN if missing
    alpha_z_fit_at_ref = evaluate_alpha_z_at(beta[b_idx], np.pi/2, map_ab)  
    if np.isnan(alpha_z_measured):
        print("Warning: measured alpha_z at (beta=0,gamma=90) is NaN — cannot compute zeta directly.")
        result['zeta'] = np.nan
    else:
        zeta = alpha_z_measured / alpha_z_fit_at_ref
        result['zeta'] = zeta

print("\nRequested Fourier coefficients (paper Table S2 style):\n")
for k in ['A0,0','A1,0','B1,1','B0,1','B-1,1','C1,1','C0,1','C-1,1','D1,0']:
    print(f"{k:6s} : {result[k]: .6f}")

print(f"\nscaling factor zeta : {result['zeta']:.6f}")
