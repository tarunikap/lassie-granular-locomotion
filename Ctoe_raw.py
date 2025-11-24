import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

folder_path = r"D:\Awesomnesss\Summer '25 Intern\USC\LASSIE\LASSIE_DataForHeatMap\LASSIE_DataForHeatMap\C toe\Cleg_d1"
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
num_files = len(csv_files)


colors = plt.cm.jet(np.linspace(0, 1, num_files))  

plt.figure(figsize=(10, 6))

for idx, filename in enumerate(sorted(csv_files)):
    file_path = os.path.join(folder_path, filename)
    df = pd.read_csv(file_path)

    position = df.iloc[:, 1]  
    current = df.iloc[:, 3]   

    
    plt.plot(position, current, color=colors[idx], label=filename)


plt.xlabel("Position")
plt.ylabel(" Current")
plt.title("Current vs Position ")
plt.legend(fontsize='small', loc='best')
plt.grid(True)
plt.tight_layout()


plt.show()
