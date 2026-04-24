#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_INPUT_CSV = Path("output/forecast_tank_predictions.csv")
OUTPUT_DIR = Path("output/forecast_plots")

TANK_COLS = [
    "tank_bottom_pred_c",
    "tank_mid_pred_c",
    "tank_mid_hi_pred_c",
    "tank_top_pred_c",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot forecast tank predictions and save figures to disk."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"Input forecast prediction CSV (default: {DEFAULT_INPUT_CSV})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.csv)
    if "timestamp" not in df.columns:
        raise ValueError("Missing required column: timestamp")

    required_cols = TANK_COLS + ["predicted_ashp_heat_kwh", "predicted_dhw_draw_energy_kwh"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values("timestamp")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 4 predicted tank temperatures
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df["timestamp"], df["tank_bottom_pred_c"], label="Bottom", linewidth=1.2)
    ax.plot(df["timestamp"], df["tank_mid_pred_c"], label="Mid", linewidth=1.2)
    ax.plot(df["timestamp"], df["tank_mid_hi_pred_c"], label="Mid Hi", linewidth=1.2)
    ax.plot(df["timestamp"], df["tank_top_pred_c"], label="Top", linewidth=1.2)
    ax.set_title("Forecast Tank Temperatures")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "forecast_tank_temperatures.png", dpi=150)
    plt.close(fig)

    # 2) Predicted ASHP heat
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["timestamp"], df["predicted_ashp_heat_kwh"], color="tab:orange", linewidth=1.1)
    ax.set_title("Forecast Predicted ASHP Heat")
    ax.set_xlabel("Time")
    ax.set_ylabel("ASHP Heat (kWh/timestep)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "forecast_predicted_ashp_heat.png", dpi=150)
    plt.close(fig)

    # 3) Predicted DHW draw energy
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["timestamp"], df["predicted_dhw_draw_energy_kwh"], color="tab:green", linewidth=1.1)
    ax.set_title("Forecast Predicted DHW Draw Energy")
    ax.set_xlabel("Time")
    ax.set_ylabel("DHW Draw Energy (kWh/timestep)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "forecast_predicted_dhw_draw_energy.png", dpi=150)
    plt.close(fig)

    print("Saved forecast plots to output/forecast_plots")
    print(f"Input CSV: {args.csv}")


if __name__ == "__main__":
    main()
