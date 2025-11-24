import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_file = r"D:\Awesomnesss\Summer '25 Intern\USC\LASSIE\LASSIE_DataForHeatMap\LASSIE_DataForHeatMap\C toe\Cleg_d1\Cleg_d1_t1.csv"
kt = 0.0973   # Nm/A (torque constant)
R = 0.05      # m, radius of C-toe
w = 0.025     # m, width (into page, contact width)
Nseg = 10     # divide trajectory into segments

df = pd.read_csv(csv_file)
pos = df["ODrive 2 Position"].to_numpy()
vel = df["ODrive 2 Velocity"].to_numpy()
cur = df["ODrive 2 Current"].to_numpy()

torque = cur * kt                  # Nm
F_total = torque / R                # N (approx tip force)
A = w * (2*R/Nseg)                  # area of each patch
stress = F_total / A                # N/m² -> stress

theta = (pos - pos.min())/(pos.max()-pos.min()) * np.pi  # 0 to pi
beta = np.linspace(-np.pi/2, np.pi/2, len(pos))          # attack angle sweep
gamma = np.linspace(-np.pi/2, np.pi/2, len(pos))         # velocity angle sweep

bins = np.arange(-90, 91, 15)
beta_deg = np.degrees(beta)
gamma_deg = np.degrees(gamma)

# scale down by 1e5
alpha_z = (stress * np.cos(beta) * np.cos(gamma)) / 1e5
alpha_x = (stress * np.cos(beta) * np.sin(gamma)) / 1e5

heatmap_z = np.zeros((len(bins)-1, len(bins)-1))
heatmap_x = np.zeros((len(bins)-1, len(bins)-1))

for b, g, az, ax in zip(beta_deg, gamma_deg, alpha_z, alpha_x):
    ib = np.digitize(b, bins)-1
    ig = np.digitize(g, bins)-1
    if 0 <= ib < len(bins)-1 and 0 <= ig < len(bins)-1:
        heatmap_z[ib, ig] += az
        heatmap_x[ib, ig] += ax

fig, axs = plt.subplots(1,2, figsize=(12,5))

im1 = axs[0].imshow(heatmap_z, origin='lower',
                    extent=[-90,90,-90,90], cmap='RdYlBu')
axs[0].set_title("αz")
axs[0].set_xlabel("γ (deg)")
axs[0].set_ylabel("β (deg)")
plt.colorbar(im1, ax=axs[0])

im2 = axs[1].imshow(heatmap_x, origin='lower',
                    extent=[-90,90,-90,90], cmap='RdYlBu')
axs[1].set_title("αx")
axs[1].set_xlabel("γ (deg)")
axs[1].set_ylabel("β (deg)")
plt.colorbar(im2, ax=axs[1])

plt.tight_layout()
plt.show()
