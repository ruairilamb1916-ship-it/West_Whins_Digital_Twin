#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/FullDS_Findhorn.csv")

if "Time" not in df.columns:
    raise ValueError("Could not find required datetime column: 'Time'.")

datetime_col = "Time"
df[datetime_col] = pd.to_datetime(df[datetime_col], errors="raise", dayfirst=True)

print(f"Using datetime column: {datetime_col}")
df = df.rename(columns={datetime_col: "timestamp"})
df = df.sort_values("timestamp").set_index("timestamp")

cols = [
    "Tank Bottom [°C]",
    "Tank Mid [°C]",
    "Tank Mid Hi [°C]",
    "Tank Top [°C]",
]

plt.figure(figsize=(15, 6))

plotted = False

for c in cols:
    if c in df.columns:
        plt.plot(df.index, df[c], label=c)
        plotted = True
    else:
        print(f"Skipping missing column: {c}")

plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.title("Tank Temperature Stratification (Full Dataset)")
if plotted:
    plt.legend()
else:
    print("No tank temperature columns found to plot.")
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = Path("output/plots/tank_temperatures.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300)
plt.show()
