import pandas as pd
import matplotlib.pyplot as plt


file_path = r"D:\Awesomnesss\Summer '25 Intern\USC\LASSIE\LASSIE_DataForHeatMap\LASSIE_DataForHeatMap\C toe\leg_d3\Cleg_d3_t1.csv"

# read CSV
df = pd.read_csv(file_path)

# pick columns (adjust indices if needed)
position = df.iloc[:, 1].to_numpy()
current = df.iloc[:, 3].to_numpy()

# subtract 0.1 from all x-values
position_shifted = position - 0.1

# Raw currents with shifted positions
plt.figure(figsize=(10, 6))
plt.plot(position_shifted, current, label="Raw Current (Shifted)", color="blue")

plt.xlabel("Position - 0.1")
plt.ylabel("Current")
plt.title("Current vs Position (Raw, Shifted)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
