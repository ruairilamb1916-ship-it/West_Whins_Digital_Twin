#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START_PROB_CSV = Path("output/dhw_stochastic_model/dhw_event_start_probability.csv")
SIZE_DIST_CSV = Path("output/dhw_stochastic_model/dhw_event_size_distribution_monthly.csv")
DURATION_DIST_CSV = Path("output/dhw_stochastic_model/dhw_event_duration_distribution_monthly.csv")
DEFAULT_OUTPUT_CSV = Path("output/dhw_stochastic_model/dhw_synthetic_profile.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a synthetic DHW draw profile from empirical stochastic tables."
    )
    parser.add_argument("--start", type=str, required=True, help="Start timestamp (e.g. 2024-01-01 00:00:00)")
    parser.add_argument("--end", type=str, required=True, help="End timestamp (e.g. 2024-12-31 23:30:00)")
    parser.add_argument(
        "--freq",
        type=str,
        required=True,
        help="Timestep frequency (e.g. 30min, 1H)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    return parser.parse_args()


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start_prob = pd.read_csv(START_PROB_CSV)
    size_dist = pd.read_csv(SIZE_DIST_CSV)
    duration_dist = pd.read_csv(DURATION_DIST_CSV)

    req_start_cols = {"month", "hour_of_day", "event_start_probability"}
    req_size_cols = {"month", "size_value"}
    req_dur_cols = {"month", "duration_min_value"}

    if not req_start_cols.issubset(start_prob.columns):
        raise ValueError(f"Missing columns in {START_PROB_CSV}: {sorted(req_start_cols)}")
    if not req_size_cols.issubset(size_dist.columns):
        raise ValueError(f"Missing columns in {SIZE_DIST_CSV}: {sorted(req_size_cols)}")
    if not req_dur_cols.issubset(duration_dist.columns):
        raise ValueError(f"Missing columns in {DURATION_DIST_CSV}: {sorted(req_dur_cols)}")

    return start_prob, size_dist, duration_dist


def build_lookup_tables(
    start_prob: pd.DataFrame,
    size_dist: pd.DataFrame,
    duration_dist: pd.DataFrame,
) -> tuple[dict[tuple[int, int], float], dict[int, np.ndarray], dict[int, np.ndarray]]:
    p_lookup: dict[tuple[int, int], float] = {}
    for _, row in start_prob.iterrows():
        month = int(row["month"])
        hour = int(row["hour_of_day"])
        p = float(row["event_start_probability"])
        p_lookup[(month, hour)] = float(np.clip(p, 0.0, 1.0))

    size_lookup: dict[int, np.ndarray] = {}
    for month in range(1, 13):
        vals = (
            pd.to_numeric(
                size_dist.loc[size_dist["month"] == month, "size_value"],
                errors="coerce",
            )
            .dropna()
            .to_numpy(dtype=float)
        )
        vals = vals[vals > 0]
        size_lookup[month] = vals

    duration_lookup: dict[int, np.ndarray] = {}
    for month in range(1, 13):
        vals = (
            pd.to_numeric(
                duration_dist.loc[duration_dist["month"] == month, "duration_min_value"],
                errors="coerce",
            )
            .dropna()
            .to_numpy(dtype=float)
        )
        vals = vals[vals > 0]
        duration_lookup[month] = vals

    return p_lookup, size_lookup, duration_lookup


def sample_from_monthly(values_by_month: dict[int, np.ndarray], month: int, fallback: float) -> float:
    vals = values_by_month.get(month, np.array([], dtype=float))
    if vals.size == 0:
        return fallback
    return float(np.random.choice(vals))


def simulate_profile(
    index: pd.DatetimeIndex,
    freq_minutes: float,
    p_lookup: dict[tuple[int, int], float],
    size_lookup: dict[int, np.ndarray],
    duration_lookup: dict[int, np.ndarray],
) -> pd.DataFrame:
    n = len(index)
    event_start = np.zeros(n, dtype=int)
    draw_active = np.zeros(n, dtype=int)
    draw_size_l = np.zeros(n, dtype=float)
    event_id = np.zeros(n, dtype=int)

    in_event = False
    remaining_steps = 0
    current_event_id = 0
    per_step_draw_l = 0.0

    for i, ts in enumerate(index):
        if in_event:
            draw_active[i] = 1
            draw_size_l[i] = per_step_draw_l
            event_id[i] = current_event_id
            remaining_steps -= 1
            if remaining_steps <= 0:
                in_event = False
            continue

        month = int(ts.month)
        hour = int(ts.hour)
        p_start = p_lookup.get((month, hour), 0.0)

        if np.random.random() < p_start:
            sampled_size_l = sample_from_monthly(size_lookup, month=month, fallback=10.0)
            sampled_duration_min = sample_from_monthly(duration_lookup, month=month, fallback=60.0)

            n_steps = max(1, int(np.round(sampled_duration_min / max(freq_minutes, 1e-6))))
            per_step_draw_l = float(sampled_size_l) / float(n_steps)

            current_event_id += 1
            event_start[i] = 1
            draw_active[i] = 1
            draw_size_l[i] = per_step_draw_l
            event_id[i] = current_event_id

            in_event = True
            remaining_steps = n_steps - 1

    return pd.DataFrame(
        {
            "timestamp": index,
            "dhw_event_start": event_start,
            "dhw_draw_active": draw_active,
            "dhw_draw_size_l": draw_size_l,
            "event_id": event_id,
        }
    )


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    start = pd.to_datetime(args.start, errors="raise")
    end = pd.to_datetime(args.end, errors="raise")
    if end < start:
        raise ValueError("--end must be after --start")

    freq_td = pd.to_timedelta(args.freq)
    freq_minutes = float(freq_td.total_seconds() / 60.0)
    if freq_minutes <= 0:
        raise ValueError("--freq must be a positive timestep")

    index = pd.date_range(start=start, end=end, freq=args.freq)
    if len(index) == 0:
        raise ValueError("Requested period and frequency produced no timesteps")

    start_prob, size_dist, duration_dist = load_tables()
    p_lookup, size_lookup, duration_lookup = build_lookup_tables(start_prob, size_dist, duration_dist)

    profile = simulate_profile(
        index=index,
        freq_minutes=freq_minutes,
        p_lookup=p_lookup,
        size_lookup=size_lookup,
        duration_lookup=duration_lookup,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(args.output, index=False)

    print("Synthetic DHW profile saved")
    print(f"Rows: {len(profile)}")
    print(f"Event starts: {int(profile['dhw_event_start'].sum())}")
    print(f"Active draw timesteps: {int(profile['dhw_draw_active'].sum())}")
    print(f"Total synthetic draw [L]: {profile['dhw_draw_size_l'].sum():.2f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
