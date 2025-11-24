import numpy as np
import pandas as pd
import math, os

def wrap_to_180(angle_deg):
    """Wrap angle to (-180, 180] degrees."""
    return ((angle_deg + 180) % 360) - 180

def motor_to_phi(motor_pos):
    """Map motor_pos ∈ [0,1] to rotation φ (radians)."""
    return 2.0 * math.pi * (motor_pos - 0.15)

def compute_toe_segments_csv(
    outfile=r"D:\Awesomnesss\Summer '25 Intern\USC\LASSIE\toe_segments_output.csv",
    N=10,
    R=0.05,
    start_motor=0.10,
    stop_motor=0.60,
    step=0.01,
    z_ground=0.0368,
    eps=1e-12
):
    """
    Compute toe divided into N arc segments for motor positions from start_motor to stop_motor.
    Save CSV with start, end, midpoint coordinates, beta/gamma, depth, and in_ground flag.
    """

    out_dir = os.path.dirname(os.path.abspath(outfile))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    dtheta = math.pi / N
    rows = []
    motor_positions = np.arange(start_motor, stop_motor + eps, step)

    for m in motor_positions:
        phi = motor_to_phi(float(m))
        for k in range(N):
            # angles
            theta_prox = k * dtheta + phi
            theta_dist = (k+1) * dtheta + phi
            theta_mid  = (k+0.5) * dtheta + phi

            # coordinates
            x_prox, z_prox = -R + R * math.cos(theta_prox), R * math.sin(theta_prox)
            x_dist, z_dist = -R + R * math.cos(theta_dist), R * math.sin(theta_dist)
            x_mid,  z_mid  = -R + R * math.cos(theta_mid),  R * math.sin(theta_mid)

            # beta: orientation of segment
            vx_seg, vz_seg = x_dist - x_prox, z_dist - z_prox
            ang_seg = math.degrees(math.atan2(vz_seg, vx_seg))
            beta_deg = wrap_to_180(180.0 - ang_seg)

            # gamma: velocity direction at midpoint
            vx_v, vz_v = -z_mid, x_mid
            ang_v = math.degrees(math.atan2(vz_v, vx_v))
            gamma_deg = wrap_to_180(ang_v - 180.0)

            # depth and in_ground (midpoint only)
            if z_mid > z_ground:
                depth_m = float(z_mid - z_ground)
                in_ground = 1
            else:
                depth_m = 0.0
                in_ground = 0

            rows.append({
                "motor_pos": round(float(m), 6),
                "segment": int(k),
                "x_prox": float(x_prox), "z_prox": float(z_prox),
                "x_dist": float(x_dist), "z_dist": float(z_dist),
                "x_mid": float(x_mid),   "z_mid": float(z_mid),
                "beta_deg": float(beta_deg),
                "gamma_deg": float(gamma_deg),
                "depth_m": float(depth_m),
                "in_ground": int(in_ground)
            })

    df = pd.DataFrame(rows, columns=[
        "motor_pos","segment",
        "x_prox","z_prox","x_dist","z_dist",
        "x_mid","z_mid",
        "beta_deg","gamma_deg","depth_m","in_ground"
    ])
    df.to_csv(outfile, index=False)  # overwrite file

    # print preview
    print(f"\n Saved {len(df)} rows to: {outfile}")
    print("\n--- Preview (first 10 rows) ---")
    print(df.head(10).to_string(index=False))

    return df


# === Example run ===
if __name__ == "__main__":
    compute_toe_segments_csv(
        outfile=r"D:\Awesomnesss\Summer '25 Intern\USC\LASSIE\toe_segments_output.csv",
        N=100,
        R=0.05,
        start_motor=0.00,
        stop_motor=0.99,
        step=0.01,
        z_ground=0.0368
    )
