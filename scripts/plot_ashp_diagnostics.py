import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
input_path = Path("output/debug_ashp_map.csv")
output_dir = Path("output/plots")
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(input_path)

# Drop NaNs
df = df.dropna(subset=["COP_fit", "COP_target"])

# --- Scatter plot: COP_fit vs COP_target ---
plt.figure()
plt.scatter(df["COP_target"], df["COP_fit"], alpha=0.3)

# Diagonal reference line
min_val = min(df["COP_target"].min(), df["COP_fit"].min())
max_val = max(df["COP_target"].max(), df["COP_fit"].max())
plt.plot([min_val, max_val], [min_val, max_val])

plt.xlabel("COP_target (measured)")
plt.ylabel("COP_fit (model)")
plt.title("ASHP COP Fit vs Target")
plt.grid()

plt.savefig(output_dir / "cop_comparison.png", dpi=300)
plt.close()

# --- Histogram comparison ---
plt.figure()
plt.hist(df["COP_target"], bins=50, alpha=0.5, label="COP_target")
plt.hist(df["COP_fit"], bins=50, alpha=0.5, label="COP_fit")

plt.xlabel("COP")
plt.ylabel("Frequency")
plt.title("COP Distribution Comparison")
plt.legend()
plt.grid()
plt.savefig(output_dir / "cop_histogram.png", dpi=300)
plt.close()
print(f"Saved plots to {output_dir}")
