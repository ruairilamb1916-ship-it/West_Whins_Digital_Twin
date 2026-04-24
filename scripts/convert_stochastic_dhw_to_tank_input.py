#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INPUT_CSV = Path("output/dhw_stochastic_model/dhw_sample_2024.csv")
DEFAULT_OUTPUT_CSV = Path("output/dhw_stochastic_model/dhw_tank_input_2024.csv")

# Physical assumptions (kept explicit for transparency):
# - 1 litre of water is approximated as 1 kg
# - specific heat capacity of water is 4.186 kJ/(kg*K)
# Conversion constant from kJ to kWh = 1/3600
WATER_CP_KJ_PER_KG_K = 4.186
KJ_TO_KWH = 1.0 / 3600.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert sampled stochastic DHW profile to tank demand input series."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"Input sampled DHW CSV (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output tank-demand CSV (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--t-hot-c",
        type=float,
        default=50.0,
        help="Hot water delivery temperature in degC (default: 50.0)",
    )
    parser.add_argument(
        "--t-mains-c",
        type=float,
        default=10.0,
        help="Mains water temperature in degC (default: 10.0)",
    )
    parser.add_argument(
        "--annual-target-kwh",
        type=float,
        default=None,
        help="Optional annual target energy in kWh used to scale dhw_draw_energy_kwh.",
    )
    parser.add_argument(
        "--annual-scaling-mode",
        type=str,
        choices=["volume-and-energy", "energy-only"],
        default="volume-and-energy",
        help=(
            "How annual scaling is applied when --annual-target-kwh is set: "
            "'volume-and-energy' scales both dhw_draw_size_l and dhw_draw_energy_kwh, "
            "while 'energy-only' preserves the previous behaviour."
        ),
    )
    return parser.parse_args()


def compute_draw_energy_kwh(draw_l: pd.Series, t_hot_c: float, t_mains_c: float) -> pd.Series:
    # Thermal withdrawal per timestep from DHW draw volume:
    # E[kWh] = volume[L] * cp[kJ/kg/K] * (T_hot - T_mains)[K] / 3600
    delta_t = max(t_hot_c - t_mains_c, 0.0)
    return draw_l.clip(lower=0.0) * WATER_CP_KJ_PER_KG_K * delta_t * KJ_TO_KWH


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)

    required_cols = {
        "timestamp",
        "dhw_draw_active",
        "dhw_draw_size_l",
        "event_id",
    }
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError("Missing required input columns: " + ", ".join(sorted(missing)))

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df["dhw_draw_size_l"] = pd.to_numeric(df["dhw_draw_size_l"], errors="coerce").fillna(0.0)
    df["dhw_draw_active"] = pd.to_numeric(df["dhw_draw_active"], errors="coerce").fillna(0).astype(int)
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").fillna(0).astype(int)

    df["dhw_draw_energy_kwh"] = compute_draw_energy_kwh(
        draw_l=df["dhw_draw_size_l"],
        t_hot_c=args.t_hot_c,
        t_mains_c=args.t_mains_c,
    )

    original_total_l = float(df["dhw_draw_size_l"].sum())
    original_total_kwh = float(df["dhw_draw_energy_kwh"].sum())
    scale_factor = 1.0
    if args.annual_target_kwh is not None:
        if original_total_kwh > 0:
            scale_factor = float(args.annual_target_kwh) / original_total_kwh
            if args.annual_scaling_mode == "volume-and-energy":
                df["dhw_draw_size_l"] = df["dhw_draw_size_l"] * scale_factor
                df["dhw_draw_energy_kwh"] = compute_draw_energy_kwh(
                    draw_l=df["dhw_draw_size_l"],
                    t_hot_c=args.t_hot_c,
                    t_mains_c=args.t_mains_c,
                )
            else:
                df["dhw_draw_energy_kwh"] = df["dhw_draw_energy_kwh"] * scale_factor
        else:
            scale_factor = 0.0

    out = df[
        [
            "timestamp",
            "dhw_draw_active",
            "dhw_draw_size_l",
            "dhw_draw_energy_kwh",
            "event_id",
        ]
    ].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print("Saved DHW tank input series")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Assumptions: T_hot={args.t_hot_c:.1f} degC, T_mains={args.t_mains_c:.1f} degC")
    if args.annual_target_kwh is not None:
        print(f"Original total draw [L]: {original_total_l:.2f}")
        print(f"Scaled total draw [L]: {out['dhw_draw_size_l'].sum():.2f}")
        print(f"Original total draw energy [kWh]: {original_total_kwh:.2f}")
        print(f"Scaled total draw energy [kWh]: {out['dhw_draw_energy_kwh'].sum():.2f}")
        print(f"Scale factor used: {scale_factor:.6f}")
    else:
        print(f"Total draw [L]: {out['dhw_draw_size_l'].sum():.2f}")
        print(f"Total draw energy [kWh]: {out['dhw_draw_energy_kwh'].sum():.2f}")


if __name__ == "__main__":
    main()
