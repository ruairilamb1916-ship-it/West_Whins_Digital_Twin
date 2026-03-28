#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_TANK_CSV = Path("output_synth_clean_v2/tank_temps_measured_vs_sim.csv")
DEFAULT_DHW_CSV = Path("output/dhw_stochastic_model/dhw_tank_input_2024_scaled_2000.csv")

SIM_NODE_COLS = [
    "tank_bottom_sim_c",
    "tank_mid_sim_c",
    "tank_mid_hi_sim_c",
    "tank_top_sim_c",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic DHW tank simulation realism metrics."
    )
    parser.add_argument("--tank-csv", type=Path, default=DEFAULT_TANK_CSV)
    parser.add_argument("--dhw-csv", type=Path, default=DEFAULT_DHW_CSV)
    return parser.parse_args()


def load_inputs(tank_csv: Path, dhw_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tank = pd.read_csv(tank_csv)
    dhw = pd.read_csv(dhw_csv)

    if "timestamp" not in tank.columns:
        raise ValueError("Missing timestamp column in tank CSV")
    if "timestamp" not in dhw.columns:
        raise ValueError("Missing timestamp column in DHW CSV")

    missing_nodes = [c for c in SIM_NODE_COLS if c not in tank.columns]
    if missing_nodes:
        raise ValueError("Missing simulated tank columns: " + ", ".join(missing_nodes))

    tank["timestamp"] = pd.to_datetime(tank["timestamp"], errors="raise")
    dhw["timestamp"] = pd.to_datetime(dhw["timestamp"], errors="raise")

    return tank.sort_values("timestamp").reset_index(drop=True), dhw.sort_values("timestamp").reset_index(drop=True)


def stratification_validity_rate(tank: pd.DataFrame) -> float:
    cond = (
        (tank["tank_top_sim_c"] >= tank["tank_mid_hi_sim_c"])
        & (tank["tank_mid_hi_sim_c"] >= tank["tank_mid_sim_c"])
        & (tank["tank_mid_sim_c"] >= tank["tank_bottom_sim_c"])
    )
    return float(cond.mean())


def node_temperature_stats(tank: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for col in SIM_NODE_COLS:
        s = pd.to_numeric(tank[col], errors="coerce")
        rows.append(
            {
                "node": col,
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "p05": float(s.quantile(0.05)),
                "p50": float(s.quantile(0.50)),
                "p95": float(s.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    tank, dhw = load_inputs(args.tank_csv, args.dhw_csv)

    strat_rate = stratification_validity_rate(tank)
    stats_df = node_temperature_stats(tank)

    print("Synthetic DHW Tank Evaluation")
    print(f"Tank CSV: {args.tank_csv}")
    print(f"DHW CSV: {args.dhw_csv}")
    print(f"Rows (tank): {len(tank)}")
    print(f"Rows (dhw): {len(dhw)}")
    print()
    print("Stratification validity")
    print(f"  Fraction valid (top >= mid_hi >= mid >= bottom): {strat_rate:.4f}")
    print()
    print("Temperature stats (simulated nodes)")
    print(stats_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
