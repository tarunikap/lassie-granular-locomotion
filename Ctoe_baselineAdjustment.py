import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff=0.05, order=3):
    b, a = butter(order, cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

folder_path = r"D:\Awesomnesss\Summer '25 Intern\USC\LASSIE\LASSIE_DataForHeatMap\LASSIE_DataForHeatMap\C toe\Cleg_d1"
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
num_files = len(csv_files)

colors = plt.cm.jet(np.linspace(0, 1, num_files))  

# Raw currents
plt.figure(figsize=(10, 6))
for idx, filename in enumerate(sorted(csv_files)):
    file_path = os.path.join(folder_path, filename)
    df = pd.read_csv(file_path)

    position = df.iloc[:, 1].to_numpy()
    current = df.iloc[:, 3].to_numpy()

    plt.plot(position, current, color=colors[idx], label=filename)

plt.xlabel("Position")
plt.ylabel("Current")
plt.title("Current vs Position (Raw)")
plt.legend(fontsize='x-small', loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

# Smoothed currents
plt.figure(figsize=(10, 6))
for idx, filename in enumerate(sorted(csv_files)):
    file_path = os.path.join(folder_path, filename)
    df = pd.read_csv(file_path)

    position = df.iloc[:, 1].to_numpy()
    current = df.iloc[:, 3].to_numpy()

    current_smooth = butter_lowpass_filter(current, cutoff=0.05, order=3)

    plt.plot(position, current_smooth, color=colors[idx], label=filename)

plt.xlabel("Position")
plt.ylabel("Current (Smoothed)")
plt.title("Current vs Position (Smoothed)")
plt.legend(fontsize='x-small', loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

# Smoothed & baseline-shifted currents (start at 0)
plt.figure(figsize=(10, 6))
for idx, filename in enumerate(sorted(csv_files)):
    file_path = os.path.join(folder_path, filename)
    df = pd.read_csv(file_path)

    position = df.iloc[:, 1].to_numpy()
    current = df.iloc[:, 3].to_numpy()

    current_smooth = butter_lowpass_filter(current, cutoff=0.05, order=3)
    current_shifted = current_smooth - current_smooth.min()

    plt.plot(position, current_shifted, color=colors[idx], label=filename)

plt.xlabel("Position")
plt.ylabel("Current (Shifted, Smoothed)")
plt.title("Current vs Position (Baseline Adjusted to 0)")
plt.legend(fontsize='x-small', loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()
