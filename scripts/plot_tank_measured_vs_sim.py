#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIM_CSV = Path("output/tank_temps_measured_vs_sim.csv")
OUT_PLOT_NAME = "tank_temps_measured_vs_sim.png"

COLUMNS = [
    "timestamp",
    "tank_bottom_sim_c",
    "tank_mid_sim_c",
    "tank_mid_hi_sim_c",
    "tank_top_sim_c",
    "tank_bottom_meas_c",
    "tank_mid_meas_c",
    "tank_mid_hi_meas_c",
    "tank_top_meas_c",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot measured vs simulated 4-node tank temperatures."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=SIM_CSV,
        help=f"Input tank comparison CSV (default: {SIM_CSV})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_csv = args.csv

    df = pd.read_csv(in_csv)
    if "timestamp" not in df.columns:
        raise ValueError("Missing simulated datetime column: timestamp")

    missing_cols = [c for c in COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            "Missing required tank comparison columns: " + ", ".join(missing_cols)
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df[COLUMNS]
    df = df.sort_values("timestamp")
    df = df.dropna(
        subset=[
            "tank_bottom_meas_c", "tank_bottom_sim_c",
            "tank_mid_meas_c", "tank_mid_sim_c",
            "tank_mid_hi_meas_c", "tank_mid_hi_sim_c",
            "tank_top_meas_c", "tank_top_sim_c",
        ]
    )

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    plot_map = [
        ("Bottom", "tank_bottom_meas_c", "tank_bottom_sim_c"),
        ("Mid", "tank_mid_meas_c", "tank_mid_sim_c"),
        ("Mid Hi", "tank_mid_hi_meas_c", "tank_mid_hi_sim_c"),
        ("Top", "tank_top_meas_c", "tank_top_sim_c"),
    ]

    all_errors = []

    for ax, (label, meas_col, sim_col) in zip(axes, plot_map):
        ax.plot(df["timestamp"], df[meas_col], label="Measured", linewidth=1.2)
        ax.plot(df["timestamp"], df[sim_col], label="Simulated", linewidth=1.2)

        err = (df[sim_col] - df[meas_col]).to_numpy(dtype=float)
        all_errors.append(err)
        rmse = np.sqrt(np.mean(err ** 2))
        ax.text(
            0.02,
            0.98,
            f"RMSE: {rmse:.2f} °C",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

        ax.set_ylabel(f"{label} (°C)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    overall_rmse = np.sqrt(np.mean(np.concatenate(all_errors) ** 2))

    axes[-1].set_xlabel("Time")
    fig.suptitle(
        f"Measured vs Simulated Tank Temperatures (Full Dataset) | Overall RMSE: {overall_rmse:.2f} °C"
    )
    fig.tight_layout()

    output_dir = os.path.dirname(args.csv)
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    output_path = os.path.join(plot_dir, "tank_temps_measured_vs_sim.png")
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    # plt.show()


if __name__ == "__main__":
    main()
