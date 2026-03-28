#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path("output/tank_temps_measured_vs_sim.csv")

REQUIRED_COLUMNS = [
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

NODE_MAP = {
    "bottom": ("tank_bottom_sim_c", "tank_bottom_meas_c"),
    "mid": ("tank_mid_sim_c", "tank_mid_meas_c"),
    "mid_hi": ("tank_mid_hi_sim_c", "tank_mid_hi_meas_c"),
    "top": ("tank_top_sim_c", "tank_top_meas_c"),
}


def _rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err ** 2)))


def _mae(err: np.ndarray) -> float:
    return float(np.mean(np.abs(err)))


def _bias(err: np.ndarray) -> float:
    return float(np.mean(err))


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")

    df_clean = df[REQUIRED_COLUMNS].dropna(axis=0, how="any")
    if df_clean.empty:
        raise ValueError("No rows remaining after dropping NaNs in required columns.")

    print(f"Rows used: {len(df_clean)}")

    all_errors = []
    for node, (sim_col, meas_col) in NODE_MAP.items():
        err = (df_clean[sim_col] - df_clean[meas_col]).to_numpy(dtype=float)
        all_errors.append(err)

        rmse = _rmse(err)
        mae = _mae(err)
        bias = _bias(err)
        print(f"{node:7s} RMSE: {rmse:.4f} C | MAE: {mae:.4f} C | Bias: {bias:.4f} C")

    err_all = np.concatenate(all_errors)
    overall_rmse = _rmse(err_all)
    print(f"overall RMSE (all nodes): {overall_rmse:.4f} C")


if __name__ == "__main__":
    main()
