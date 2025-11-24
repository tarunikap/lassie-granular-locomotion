import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import itertools

def compute_depth_cm(y_series):
    return (y_series.iloc[0] - y_series) * 100

def plot_force_vs_computed_depth(folder_path, save_path):
    fx_slopes, fy_slopes = [], []

    os.makedirs(save_path, exist_ok=True)

    folder_suffix = os.path.basename(folder_path).replace('fx_', '').replace('fy_', '')

    # Fx
    plt.figure(figsize=(12, 7))
    fx_colors = itertools.cycle(plt.cm.tab10.colors)

    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(path)

                if 'toe_position_y' in df.columns and 'toeforce_x' in df.columns:
                    df_clean = df.copy()
                    df_clean['depth'] = compute_depth_cm(df_clean['toe_position_y'])

                    color = next(fx_colors)
                    plt.plot(df_clean['depth'], df_clean['toeforce_x'], color=color, linewidth=0.8, alpha=1.0)

                    slope, intercept, *_ = linregress(df_clean['depth'], df_clean['toeforce_x'])
                    fx_slopes.append(slope)
                    plt.plot(df_clean['depth'], slope * df_clean['depth'] + intercept,
                             linestyle='--', color=color, linewidth=0.8, alpha=1.0)
                else:
                    print(f"Skipping {file} — missing 'toe_position_y' or 'toeforce_x'")
            except Exception as e:
                print(f"Error in Fx for {file}: {e}")

    plt.title("Fx vs Depth (Penetration)", fontsize=30)
    plt.xlabel("Depth (cm)", fontsize=24)
    plt.ylabel("Fx (N)", fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.box(False)
    if fx_slopes:
        fx_mean = np.mean(fx_slopes)
        fx_std = np.std(fx_slopes)
        plt.text(0.95, 0.05, f"Mean slope: {fx_mean:.4f}\nStd dev: {fx_std:.4f}",
                 transform=plt.gca().transAxes, ha='right',
                 fontsize=25,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        print(f"\nFx mean slope = {fx_mean:.4f} ± {fx_std:.4f} N/cm²")

    fx_save_path = os.path.join(save_path, f"fx_{folder_suffix}.png")
    plt.tight_layout()
    plt.savefig(fx_save_path, dpi=300)
    plt.show()
    print(f"Fx plot saved at: {fx_save_path}")

    # Fy
    plt.figure(figsize=(12, 7))
    fy_colors = itertools.cycle(plt.cm.tab10.colors)

    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(path)

                if 'toe_position_y' in df.columns and 'toeforce_y' in df.columns:
                    df_clean = df.copy()
                    df_clean['depth'] = compute_depth_cm(df_clean['toe_position_y'])

                    color = next(fy_colors)
                    plt.plot(df_clean['depth'], df_clean['toeforce_y'], color=color, linewidth=0.8, alpha=1.0)

                    slope, intercept, *_ = linregress(df_clean['depth'], df_clean['toeforce_y'])
                    fy_slopes.append(slope)
                    plt.plot(df_clean['depth'], slope * df_clean['depth'] + intercept,
                             linestyle='--', color=color, linewidth=0.8, alpha=1.0)
                else:
                    print(f"Skipping {file} — missing 'toe_position_y' or 'toeforce_y'")
            except Exception as e:
                print(f"Error in Fy for {file}: {e}")

    plt.title("Fz vs Depth (Penetration)", fontsize=30)
    plt.xlabel("Depth (cm)", fontsize=24)
    plt.ylabel("Fz (N)", fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.box(False)
    if fy_slopes:
        fy_mean = np.mean(fy_slopes)
        fy_std = np.std(fy_slopes)
        plt.text(0.95, 0.05, f"Mean slope: {fy_mean:.4f}\nStd dev: {fy_std:.4f}",
                 transform=plt.gca().transAxes, ha='right',
                 fontsize=25,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        print(f"Fz mean slope = {fy_mean:.4f} ± {fy_std:.4f} N/cm²")

    fy_save_path = os.path.join(save_path, f"fy_{folder_suffix}.png")
    plt.tight_layout()
    plt.savefig(fy_save_path, dpi=300)
    plt.show()
    print(f"Fy plot saved at: {fy_save_path}")

# === Run ===
data_folder = r"C:\Users\Tarunika P\Desktop\Summer '25 Intern\USC\LASSIE\LASSIE_DataForHeatMap\LASSIE_DataForHeatMap\cut files\cut_plate1\cut_penetration_plate1ex_angle45"
save_folder = r"C:\Users\Tarunika P\Desktop\Summer '25 Intern\USC\LASSIE\LASSIE_DataForHeatMap\LASSIE_DataForHeatMap\plots\plot_plate75"

plot_force_vs_computed_depth(data_folder, save_folder)
