#!/usr/bin/env python3
"""
Build a stochastic DHW demand model from empirical draw events.

Inputs: dhw_draw_events.csv

Outputs:
- dhw_event_start_probability.csv       (month x hour grid of event start probabilities)
- dhw_event_size_distribution_monthly.csv  (month x size percentiles)
- dhw_event_duration_distribution_monthly.csv (month x duration percentiles)
- dhw_stochastic_model.csv              (serialized model metadata)

Includes a sampling function that generates synthetic DHW events for
a given time period using empirical distributions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EVENT_CSV = Path("output/dhw_demand_profile/dhw_draw_events.csv")
OUTPUT_DIR = Path("output/dhw_stochastic_model")

# Percentiles to compute for empirical distributions
PERCENTILES = [0, 5, 10, 25, 50, 75, 90, 95, 100]

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400


def load_events(csv_path: Path) -> pd.DataFrame:
    """Load and validate event CSV."""
    df = pd.read_csv(csv_path)
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df = df.dropna(subset=["start_time"])
    return df.sort_values("start_time").reset_index(drop=True)


def build_event_start_probability(events: pd.DataFrame) -> pd.DataFrame:
    """
    Build P(event starts | month, hour) as observed frequency.

    For each (month, hour) bin, compute:
      P_start = n_events_starting / n_possible_timesteps

    Assumes 30-minute timesteps.
    """
    rows: list[dict] = []

    for month in range(1, 13):
        month_events = events[events["month"] == month]
        if month_events.empty:
            days_in_month = 0
        else:
            year = month_events["start_time"].dt.year.iloc[0]
            month_df = pd.date_range(start=f"{year}-{month:02d}-01", periods=1, freq="MS")
            if month_df.month[-1] == 12:
                next_month = pd.date_range(start=f"{year + 1}-01-01", periods=1)
            else:
                next_month = pd.date_range(start=f"{year}-{month + 1:02d}-01", periods=1)
            days_in_month = (next_month[0] - month_df[0]).days

        timesteps_per_day = 48  # 30-minute resolution
        total_timesteps_in_month = days_in_month * timesteps_per_day

        for hour in range(24):
            hour_events = month_events[month_events["hour_of_day"] == hour]
            n_starts = len(hour_events)
            # Assume 1 timestep per hour (simplified approximation)
            # In reality, there are ~2 per hour at 30-min resolution
            # but we count actual starts per hour bin observed
            timesteps_this_hour = max(1, total_timesteps_in_month // 24)

            p_start = float(n_starts) / float(timesteps_this_hour) if timesteps_this_hour > 0 else 0.0

            rows.append(
                {
                    "month": month,
                    "hour_of_day": hour,
                    "n_events_observed": n_starts,
                    "event_start_probability": p_start,
                }
            )

    return pd.DataFrame(rows)


def build_monthly_distributions(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute empirical percentiles for event size and duration by month.
    
    Returns:
        (size_dist_df, duration_dist_df)
    """
    size_rows: list[dict] = []
    duration_rows: list[dict] = []

    for month in range(1, 13):
        month_events = events[events["month"] == month]

        if not month_events.empty:
            sizes = month_events["event_size"].to_numpy(dtype=float)
            durations = month_events["duration_min"].to_numpy(dtype=float)

            for pct in PERCENTILES:
                size_rows.append(
                    {
                        "month": month,
                        "percentile": pct,
                        "size_value": float(np.percentile(sizes, pct)),
                    }
                )
                duration_rows.append(
                    {
                        "month": month,
                        "percentile": pct,
                        "duration_min_value": float(np.percentile(durations, pct)),
                    }
                )
        else:
            # No events in this month; use NaN for all percentiles
            for pct in PERCENTILES:
                size_rows.append(
                    {
                        "month": month,
                        "percentile": pct,
                        "size_value": np.nan,
                    }
                )
                duration_rows.append(
                    {
                        "month": month,
                        "percentile": pct,
                        "duration_min_value": np.nan,
                    }
                )

    return pd.DataFrame(size_rows), pd.DataFrame(duration_rows)


def build_model_metadata(events: pd.DataFrame) -> pd.DataFrame:
    """Store model metadata: date range, total events, signal type, etc."""
    metadata = {
        "attribute": [
            "total_events",
            "date_min",
            "date_max",
            "months_with_events",
            "hours_with_events",
            "mean_events_per_day",
            "signal_type",
        ],
        "value": [
            len(events),
            str(events["start_time"].min()),
            str(events["start_time"].max()),
            len(events["month"].unique()),
            len(events["hour_of_day"].unique()),
            round(len(events) / ((events["start_time"].max() - events["start_time"].min()).days), 2),
            "ST Flow [L] with optional cooling proxy",
        ],
    }
    return pd.DataFrame(metadata)


class StochasticDHWModel:
    """Stochastic DHW demand generator based on empirical event distributions."""

    def __init__(
        self,
        start_probability: pd.DataFrame,
        size_distribution: pd.DataFrame,
        duration_distribution: pd.DataFrame,
        random_seed: int | None = None,
    ):
        """
        Initialize model from distributions.

        Parameters
        ----------
        start_probability : pd.DataFrame
            Month x Hour grid of P(event starts).
        size_distribution : pd.DataFrame
            Month x Percentile table of event sizes.
        duration_distribution : pd.DataFrame
            Month x Percentile table of event durations.
        random_seed : int, optional
            Random seed for reproducibility.
        """
        self.start_prob = start_probability
        self.size_dist = size_distribution
        self.duration_dist = duration_distribution
        if random_seed is not None:
            np.random.seed(random_seed)

    def sample_event_size(self, month: int) -> float:
        """Sample an event size from monthly empirical distribution."""
        month_data = self.size_dist[self.size_dist["month"] == month]
        if month_data.empty:
            return 10.0  # default fallback
        sizes = month_data["size_value"].dropna().to_numpy(dtype=float)
        if len(sizes) == 0:
            return 10.0
        return float(np.random.choice(sizes))

    def sample_event_duration(self, month: int) -> float:
        """Sample an event duration from monthly empirical distribution."""
        month_data = self.duration_dist[self.duration_dist["month"] == month]
        if month_data.empty:
            return 60.0  # default fallback in minutes
        durations = month_data["duration_min_value"].dropna().to_numpy(dtype=float)
        if len(durations) == 0:
            return 60.0
        return float(np.random.choice(durations))

    def get_event_start_probability(self, month: int, hour: int) -> float:
        """Get P(event starts | month, hour)."""
        row = self.start_prob[(self.start_prob["month"] == month) & (self.start_prob["hour_of_day"] == hour)]
        if row.empty:
            return 0.0
        return float(row["event_start_probability"].iloc[0])

    def generate_synthetic_events(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        timestep_minutes: int = 30,
    ) -> pd.DataFrame:
        """
        Generate synthetic DHW events for a time period.

        Parameters
        ----------
        start_time : pd.Timestamp
            Start of period.
        end_time : pd.Timestamp
            End of period.
        timestep_minutes : int
            Simulation timestep (default 30 min).

        Returns
        -------
        pd.DataFrame
            Columns: timestamp, event_size, duration_min
            Each row is a potential event start with sampled properties.
        """
        timestamps = pd.date_range(start=start_time, end=end_time, freq=f"{timestep_minutes}min")

        events: list[dict] = []

        for ts in timestamps:
            month = ts.month
            hour = ts.hour

            p_start = self.get_event_start_probability(month, hour)

            if np.random.uniform() < p_start:
                size = self.sample_event_size(month)
                duration = self.sample_event_duration(month)
                events.append(
                    {
                        "timestamp": ts,
                        "month": month,
                        "hour_of_day": hour,
                        "event_size": size,
                        "duration_min": duration,
                    }
                )

        return pd.DataFrame(events)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build stochastic DHW demand model from empirical events."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=EVENT_CSV,
        help=f"Path to dhw_draw_events.csv (default: {EVENT_CSV}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for model CSVs (default: {OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading events from {args.input_csv}...")
    events = load_events(args.input_csv)
    print(f"Loaded {len(events)} events")

    print("Building event start probability table...")
    start_prob = build_event_start_probability(events)

    print("Building empirical distributions...")
    size_dist, duration_dist = build_monthly_distributions(events)

    print("Building model metadata...")
    metadata = build_model_metadata(events)

    # Save outputs
    start_prob.to_csv(args.output_dir / "dhw_event_start_probability.csv", index=False)
    size_dist.to_csv(args.output_dir / "dhw_event_size_distribution_monthly.csv", index=False)
    duration_dist.to_csv(
        args.output_dir / "dhw_event_duration_distribution_monthly.csv", index=False
    )
    metadata.to_csv(args.output_dir / "dhw_stochastic_model_metadata.csv", index=False)

    print("Outputs saved:")
    print(f"  {args.output_dir / 'dhw_event_start_probability.csv'}")
    print(f"  {args.output_dir / 'dhw_event_size_distribution_monthly.csv'}")
    print(f"  {args.output_dir / 'dhw_event_duration_distribution_monthly.csv'}")
    print(f"  {args.output_dir / 'dhw_stochastic_model_metadata.csv'}")
    print()

    # Example: generate synthetic events
    print("Generating sample synthetic events (Jan 1–7, 2024)...")
    model = StochasticDHWModel(start_prob, size_dist, duration_dist, random_seed=42)
    sample_start = pd.Timestamp("2024-01-01 00:00:00")
    sample_end = pd.Timestamp("2024-01-07 23:30:00")
    synthetic = model.generate_synthetic_events(sample_start, sample_end, timestep_minutes=30)
    print(f"Generated {len(synthetic)} synthetic event starts over 7 days")
    if not synthetic.empty:
        print("Sample synthetic events:")
        print(synthetic.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
