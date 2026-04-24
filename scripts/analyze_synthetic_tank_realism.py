#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TANK_CSV = Path("output_synth_clean_v2/tank_temps_measured_vs_sim.csv")
DEFAULT_DHW_CSV = Path("output/dhw_stochastic_model/dhw_tank_input_2024_scaled_2000.csv")
DEFAULT_OUTPUT_DIR = Path("output_synth_clean_v2/analysis")

SIM_NODE_COLS = [
    "tank_bottom_sim_c",
    "tank_mid_sim_c",
    "tank_mid_hi_sim_c",
    "tank_top_sim_c",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate physical realism of synthetic-demand tank simulation runs."
    )
    parser.add_argument("--tank-csv", type=Path, default=DEFAULT_TANK_CSV)
    parser.add_argument("--dhw-csv", type=Path, default=DEFAULT_DHW_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lower-bound-c", type=float, default=10.0)
    parser.add_argument("--upper-bound-c", type=float, default=70.0)
    return parser.parse_args()


def infer_timestep_minutes(ts: pd.Series) -> float:
    diffs = ts.sort_values().diff().dropna()
    if diffs.empty:
        return 0.0
    return float(diffs.dt.total_seconds().median() / 60.0)


def load_inputs(tank_csv: Path, dhw_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tank = pd.read_csv(tank_csv)
    dhw = pd.read_csv(dhw_csv)

    if "timestamp" not in tank.columns:
        raise ValueError("Missing timestamp in tank CSV")
    if "timestamp" not in dhw.columns:
        raise ValueError("Missing timestamp in DHW CSV")

    for col in SIM_NODE_COLS:
        if col not in tank.columns:
            raise ValueError(f"Missing simulated node column in tank CSV: {col}")

    for col in ["dhw_draw_active", "dhw_draw_size_l", "dhw_draw_energy_kwh", "event_id"]:
        if col not in dhw.columns:
            raise ValueError(f"Missing DHW column in DHW CSV: {col}")

    tank["timestamp"] = pd.to_datetime(tank["timestamp"], errors="raise")
    dhw["timestamp"] = pd.to_datetime(dhw["timestamp"], errors="raise")

    dhw["dhw_draw_active"] = pd.to_numeric(dhw["dhw_draw_active"], errors="coerce").fillna(0).astype(int)
    dhw["dhw_draw_size_l"] = pd.to_numeric(dhw["dhw_draw_size_l"], errors="coerce").fillna(0.0)
    dhw["dhw_draw_energy_kwh"] = pd.to_numeric(dhw["dhw_draw_energy_kwh"], errors="coerce").fillna(0.0)
    dhw["event_id"] = pd.to_numeric(dhw["event_id"], errors="coerce").fillna(0).astype(int)

    return tank.sort_values("timestamp"), dhw.sort_values("timestamp")


def compute_realism_metrics(
    merged: pd.DataFrame,
    lower_bound_c: float,
    upper_bound_c: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    top = merged["tank_top_sim_c"]
    mid_hi = merged["tank_mid_hi_sim_c"]
    mid = merged["tank_mid_sim_c"]
    bottom = merged["tank_bottom_sim_c"]

    strat_ok = (top >= mid_hi) & (mid_hi >= mid) & (mid >= bottom)
    strat_rate = float(strat_ok.mean()) if len(strat_ok) else float("nan")

    node_rows: list[dict] = []
    for col in SIM_NODE_COLS:
        s = pd.to_numeric(merged[col], errors="coerce")
        outside = (s < lower_bound_c) | (s > upper_bound_c)
        node_rows.append(
            {
                "node": col,
                "q05_c": float(s.quantile(0.05)),
                "q50_c": float(s.quantile(0.50)),
                "q95_c": float(s.quantile(0.95)),
                "min_c": float(s.min()),
                "max_c": float(s.max()),
                "outside_bounds_fraction": float(outside.mean()),
            }
        )

    node_df = pd.DataFrame(node_rows)
    overall_outside = (
        (merged[SIM_NODE_COLS] < lower_bound_c) | (merged[SIM_NODE_COLS] > upper_bound_c)
    ).any(axis=1)

    summary = {
        "stratification_validity_rate": strat_rate,
        "outside_bounds_fraction_any_node": float(overall_outside.mean()),
    }
    return summary, node_df


def compute_event_response_metrics(merged: pd.DataFrame, timestep_minutes: float) -> dict[str, float]:
    active = merged["dhw_draw_active"].fillna(0).astype(int) > 0
    if not active.any():
        return {
            "n_events": 0,
            "avg_bottom_drop_c": float("nan"),
            "median_bottom_drop_c": float("nan"),
            "avg_top_drop_c": float("nan"),
            "median_top_drop_c": float("nan"),
            "avg_top_recovery_min": float("nan"),
            "median_top_recovery_min": float("nan"),
        }

    starts = np.where(active & ~active.shift(1, fill_value=False))[0]
    ends = np.where(active & ~active.shift(-1, fill_value=False))[0]

    bottom_drops: list[float] = []
    top_drops: list[float] = []
    top_recovery_minutes: list[float] = []

    top_series = merged["tank_top_sim_c"].to_numpy(dtype=float)
    bottom_series = merged["tank_bottom_sim_c"].to_numpy(dtype=float)

    for start_idx, end_idx in zip(starts, ends):
        bottom_pre = float(bottom_series[start_idx])
        top_pre = float(top_series[start_idx])

        bottom_min = float(np.nanmin(bottom_series[start_idx : end_idx + 1]))
        top_min = float(np.nanmin(top_series[start_idx : end_idx + 1]))

        bottom_drops.append(bottom_pre - bottom_min)
        top_drops.append(top_pre - top_min)

        recover_idx = None
        for j in range(end_idx + 1, len(top_series)):
            if np.isfinite(top_series[j]) and top_series[j] >= top_pre:
                recover_idx = j
                break

        if recover_idx is not None and timestep_minutes > 0:
            top_recovery_minutes.append(float((recover_idx - end_idx) * timestep_minutes))

    return {
        "n_events": int(len(starts)),
        "avg_bottom_drop_c": float(np.nanmean(bottom_drops)) if bottom_drops else float("nan"),
        "median_bottom_drop_c": float(np.nanmedian(bottom_drops)) if bottom_drops else float("nan"),
        "avg_top_drop_c": float(np.nanmean(top_drops)) if top_drops else float("nan"),
        "median_top_drop_c": float(np.nanmedian(top_drops)) if top_drops else float("nan"),
        "avg_top_recovery_min": float(np.nanmean(top_recovery_minutes)) if top_recovery_minutes else float("nan"),
        "median_top_recovery_min": float(np.nanmedian(top_recovery_minutes)) if top_recovery_minutes else float("nan"),
    }


def compute_dhw_energy_metrics(dhw: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    dhw_local = dhw.copy()
    dhw_local["month"] = dhw_local["timestamp"].dt.to_period("M").astype(str)
    monthly = (
        dhw_local.groupby("month", as_index=False)["dhw_draw_energy_kwh"]
        .sum()
        .rename(columns={"dhw_draw_energy_kwh": "monthly_dhw_energy_kwh"})
    )
    annual_total = float(dhw_local["dhw_draw_energy_kwh"].sum())
    return annual_total, monthly


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tank, dhw = load_inputs(args.tank_csv, args.dhw_csv)

    merged = tank.merge(
        dhw[["timestamp", "dhw_draw_active", "dhw_draw_size_l", "dhw_draw_energy_kwh", "event_id"]],
        on="timestamp",
        how="left",
    )
    merged["dhw_draw_active"] = merged["dhw_draw_active"].fillna(0).astype(int)
    merged["dhw_draw_size_l"] = merged["dhw_draw_size_l"].fillna(0.0)
    merged["dhw_draw_energy_kwh"] = merged["dhw_draw_energy_kwh"].fillna(0.0)
    merged["event_id"] = merged["event_id"].fillna(0).astype(int)

    timestep_minutes = infer_timestep_minutes(merged["timestamp"])

    realism_summary, node_stats = compute_realism_metrics(
        merged,
        lower_bound_c=args.lower_bound_c,
        upper_bound_c=args.upper_bound_c,
    )
    event_metrics = compute_event_response_metrics(merged, timestep_minutes=timestep_minutes)
    annual_dhw_kwh, monthly_energy = compute_dhw_energy_metrics(dhw)

    summary_rows = [
        {"metric": "stratification_validity_rate", "value": realism_summary["stratification_validity_rate"]},
        {"metric": "outside_bounds_fraction_any_node", "value": realism_summary["outside_bounds_fraction_any_node"]},
        {"metric": "annual_dhw_energy_kwh", "value": annual_dhw_kwh},
        {"metric": "n_events", "value": event_metrics["n_events"]},
        {"metric": "avg_bottom_drop_c", "value": event_metrics["avg_bottom_drop_c"]},
        {"metric": "median_bottom_drop_c", "value": event_metrics["median_bottom_drop_c"]},
        {"metric": "avg_top_drop_c", "value": event_metrics["avg_top_drop_c"]},
        {"metric": "median_top_drop_c", "value": event_metrics["median_top_drop_c"]},
        {"metric": "avg_top_recovery_min", "value": event_metrics["avg_top_recovery_min"]},
        {"metric": "median_top_recovery_min", "value": event_metrics["median_top_recovery_min"]},
        {"metric": "timestep_minutes", "value": timestep_minutes},
    ]
    summary_df = pd.DataFrame(summary_rows)

    summary_csv = args.output_dir / "synthetic_tank_realism_summary.csv"
    node_csv = args.output_dir / "synthetic_tank_node_temperature_stats.csv"
    monthly_csv = args.output_dir / "synthetic_dhw_monthly_energy.csv"

    summary_df.to_csv(summary_csv, index=False)
    node_stats.to_csv(node_csv, index=False)
    monthly_energy.to_csv(monthly_csv, index=False)

    print("Synthetic-demand tank realism analysis complete")
    print(f"Tank CSV: {args.tank_csv}")
    print(f"DHW CSV: {args.dhw_csv}")
    print(f"Stratification validity rate: {realism_summary['stratification_validity_rate']:.3f}")
    print(f"Outside-bounds fraction (any node): {realism_summary['outside_bounds_fraction_any_node']:.3f}")
    print(f"Annual DHW energy [kWh]: {annual_dhw_kwh:.2f}")
    print(f"Events analyzed: {event_metrics['n_events']}")
    print(f"Avg top recovery time [min]: {event_metrics['avg_top_recovery_min']:.2f}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {node_csv}")
    print(f"Saved: {monthly_csv}")


if __name__ == "__main__":
    main()
