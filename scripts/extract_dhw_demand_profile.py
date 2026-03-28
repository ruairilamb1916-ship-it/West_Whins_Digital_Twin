#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_INPUT_CSV = Path("data/FullDS_Findhorn_clean.csv")
DEFAULT_OUTPUT_DIR = Path("output/dhw_demand_profile")
DEFAULT_TIME_COL = "Time"

# Priority order for simple, transparent signal selection.
SIGNAL_CANDIDATES = [
    "tank_bottom_c",
    "Tank Bottom [°C]",
    "Tank Bottom [�C]",
    "tank_mid_c",
    "Tank Mid [°C]",
    "Tank Mid [�C]",
    "tank_top_c",
    "Tank Top [°C]",
    "Tank Top [�C]",
]

BOTTOM_TEMP_CANDIDATES = ["tank_bottom_c", "Tank Bottom [°C]", "Tank Bottom [�C]"]
TOP_TEMP_CANDIDATES = ["tank_top_c", "Tank Top [°C]", "Tank Top [�C]"]
MID_TEMP_CANDIDATES = ["tank_mid_c", "Tank Mid [°C]", "Tank Mid [�C]"]


def read_csv_with_encoding_fallback(csv_path: Path) -> tuple[pd.DataFrame, str]:
    encodings = ["utf-8", "cp1252", "latin-1"]
    for encoding in encodings:
        try:
            return pd.read_csv(csv_path, encoding=encoding), encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "utf-8",
        b"",
        0,
        1,
        f"Failed to decode {csv_path} with encodings: {encodings}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract DHW draw events and seasonal demand profile from cleaned CSV data."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"Path to cleaned dataset CSV (default: {DEFAULT_INPUT_CSV}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for CSV tables and plots (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default=DEFAULT_TIME_COL,
        help=f"Timestamp column name (default: {DEFAULT_TIME_COL}).",
    )
    parser.add_argument(
        "--signal-col",
        type=str,
        default="auto",
        help="Signal column for event detection; use 'auto' to pick from known candidates.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Event threshold on signal. If omitted, a simple default is inferred.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of bins for event-size histograms (default: 20).",
    )
    return parser.parse_args()


def select_signal_column(df: pd.DataFrame, signal_col_arg: str) -> str:
    if signal_col_arg != "auto":
        if signal_col_arg not in df.columns:
            raise ValueError(f"Signal column not found: {signal_col_arg}")
        return signal_col_arg

    for col in BOTTOM_TEMP_CANDIDATES:
        if col in df.columns:
            return col

    for col in SIGNAL_CANDIDATES:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not auto-select signal column. Pass --signal-col explicitly."
    )


def default_threshold_for(signal_col: str) -> float:
    col_l = signal_col.lower()
    if "[l" in col_l or "flow" in col_l or col_l.endswith("_l"):
        return 1.0
    if "kwh" in col_l or "energy" in col_l:
        return 0.05
    return 0.0


def infer_unit(signal_col: str) -> str:
    col_l = signal_col.lower()
    if signal_col in {"temp_drop_proxy", "bottom_temp_drop_proxy"}:
        return "degC-step"
    if "[l" in col_l or "flow" in col_l or col_l.endswith("_l"):
        return "L"
    if "kwh" in col_l or "energy" in col_l:
        return "kWh"
    return "signal-units"


def first_present_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def smooth_series(series: pd.Series, window: int = 3) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).median()


def percentile_threshold(values: pd.Series, percentile: float, minimum: float) -> float:
    vals = pd.to_numeric(values, errors="coerce").fillna(0.0)
    vals = vals[vals > 0]
    if vals.empty:
        return minimum
    return float(max(minimum, np.percentile(vals.to_numpy(dtype=float), percentile)))


def compute_timestep_minutes(times: pd.Series) -> float:
    diffs = times.sort_values().diff().dropna()
    if diffs.empty:
        return 0.0
    return float(diffs.dt.total_seconds().median() / 60.0)


def build_events(
    df: pd.DataFrame,
    time_col: str,
    signal_col: str,
    threshold: float | None,
) -> tuple[pd.DataFrame, str, float]:
    timestep_min = compute_timestep_minutes(df[time_col])
    bottom_col = signal_col if signal_col in df.columns else first_present_column(df, BOTTOM_TEMP_CANDIDATES)
    top_col = first_present_column(df, TOP_TEMP_CANDIDATES)
    mid_col = first_present_column(df, MID_TEMP_CANDIDATES)

    if bottom_col is None:
        raise ValueError("Could not identify a bottom tank temperature column for DHW draw detection.")

    bottom = smooth_series(
        pd.to_numeric(df[bottom_col], errors="coerce").interpolate(limit_direction="both")
    )
    bottom_cooling = smooth_series((-bottom.diff()).clip(lower=0), window=3).fillna(0.0)

    support_terms = []
    if mid_col is not None:
        mid = smooth_series(
            pd.to_numeric(df[mid_col], errors="coerce").interpolate(limit_direction="both")
        )
        support_terms.append((-mid.diff()).clip(lower=0))
    if top_col is not None:
        top = smooth_series(
            pd.to_numeric(df[top_col], errors="coerce").interpolate(limit_direction="both")
        )
        support_terms.append((-top.diff()).clip(lower=0))

    if support_terms:
        support_cooling = smooth_series(
            pd.concat(support_terms, axis=1).max(axis=1).fillna(0.0),
            window=3,
        )
    else:
        support_cooling = pd.Series(np.zeros(len(df), dtype=float), index=df.index)

    bottom_threshold = percentile_threshold(bottom_cooling, percentile=92.0, minimum=0.05)
    support_threshold = percentile_threshold(support_cooling, percentile=92.0, minimum=0.03)

    if threshold is not None:
        bottom_threshold = max(float(threshold), 0.05)

    bottom_active = bottom_cooling > bottom_threshold
    support_active = support_cooling > support_threshold

    active = bottom_active | ((bottom_cooling > 0.7 * bottom_threshold) & support_active)
    signal = bottom_cooling
    used_signal = "bottom_temp_drop_proxy"
    used_threshold = bottom_threshold

    if not active.any():
        empty = pd.DataFrame(
            columns=[
                "event_id",
                "start_time",
                "end_time",
                "duration_min",
                "n_steps",
                "event_size",
                "day_of_year",
                "month",
                "hour_of_day",
            ]
        )
        return empty, used_signal, used_threshold

    start_flags = active & ~active.shift(1, fill_value=False)
    event_id_full = start_flags.cumsum()

    active_df = df.loc[active, [time_col]].copy()
    active_df["signal"] = signal.loc[active].to_numpy(dtype=float)
    active_df["event_id"] = event_id_full.loc[active].to_numpy(dtype=int)

    events = (
        active_df.groupby("event_id", as_index=False)
        .agg(
            start_time=(time_col, "min"),
            end_time=(time_col, "max"),
            n_steps=("signal", "size"),
            event_size=("signal", "sum"),
        )
        .sort_values("start_time")
        .reset_index(drop=True)
    )

    events["duration_min"] = events["n_steps"] * timestep_min
    events["day_of_year"] = events["start_time"].dt.dayofyear
    events["month"] = events["start_time"].dt.month
    events["hour_of_day"] = events["start_time"].dt.hour

    return (
        events[
            [
                "event_id",
                "start_time",
                "end_time",
                "duration_min",
                "n_steps",
                "event_size",
                "day_of_year",
                "month",
                "hour_of_day",
            ]
        ],
        used_signal,
        used_threshold,
    )


def build_monthly_size_distribution(events: pd.DataFrame, bins: int) -> pd.DataFrame:
    rows: list[dict] = []
    if events.empty:
        return pd.DataFrame(
            columns=["month", "bin_left", "bin_right", "count", "probability"]
        )

    max_size = float(events["event_size"].max())
    if max_size <= 0:
        bin_edges = np.array([0.0, 1.0], dtype=float)
    else:
        bin_edges = np.linspace(0.0, max_size, max(2, bins + 1))

    for month in range(1, 13):
        month_sizes = events.loc[events["month"] == month, "event_size"].to_numpy(dtype=float)
        counts, edges = np.histogram(month_sizes, bins=bin_edges)
        total = counts.sum()

        for i, count in enumerate(counts):
            prob = (float(count) / float(total)) if total > 0 else 0.0
            rows.append(
                {
                    "month": month,
                    "bin_left": float(edges[i]),
                    "bin_right": float(edges[i + 1]),
                    "count": int(count),
                    "probability": prob,
                }
            )

    return pd.DataFrame(rows)


def build_monthly_frequency(events: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        events.groupby("month", as_index=False)
        .agg(
            event_count=("event_id", "count"),
            mean_event_size=("event_size", "mean"),
            median_event_size=("event_size", "median"),
            total_event_size=("event_size", "sum"),
        )
    )
    all_months = pd.DataFrame({"month": np.arange(1, 13, dtype=int)})
    monthly = all_months.merge(monthly, on="month", how="left").fillna(0.0)
    monthly["event_count"] = monthly["event_count"].astype(int)
    return monthly


def build_hourly_distribution(events: pd.DataFrame) -> pd.DataFrame:
    hourly = (
        events.groupby("hour_of_day", as_index=False)
        .agg(
            event_count=("event_id", "count"),
            mean_event_size=("event_size", "mean"),
            total_event_size=("event_size", "sum"),
        )
    )
    all_hours = pd.DataFrame({"hour_of_day": np.arange(0, 24, dtype=int)})
    hourly = all_hours.merge(hourly, on="hour_of_day", how="left").fillna(0.0)
    hourly["event_count"] = hourly["event_count"].astype(int)
    return hourly


def plot_monthly_size_histograms(events: pd.DataFrame, out_path: Path, unit: str, bins: int) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    if events.empty:
        for month, ax in enumerate(axes, start=1):
            ax.set_title(f"Month {month}")
            ax.text(0.5, 0.5, "No events", ha="center", va="center", transform=ax.transAxes)
            ax.grid(True, alpha=0.2)
    else:
        max_size = float(events["event_size"].max())
        if max_size <= 0:
            bin_edges = np.array([0.0, 1.0], dtype=float)
        else:
            bin_edges = np.linspace(0.0, max_size, max(2, bins + 1))

        for month, ax in enumerate(axes, start=1):
            vals = events.loc[events["month"] == month, "event_size"].to_numpy(dtype=float)
            if len(vals) > 0:
                ax.hist(vals, bins=bin_edges, color="steelblue", alpha=0.8)
            else:
                ax.text(0.5, 0.5, "No events", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Month {month}")
            ax.grid(True, alpha=0.2)

    fig.suptitle("DHW Draw Event Size Distribution by Month")
    fig.supxlabel(f"Event size ({unit})")
    fig.supylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_monthly_frequency(monthly_freq: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(monthly_freq["month"], monthly_freq["event_count"], color="darkorange", alpha=0.85)
    ax.set_title("DHW Draw Event Frequency by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Event count")
    ax.set_xticks(np.arange(1, 13, dtype=int))
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_hourly_distribution(hourly: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(hourly["hour_of_day"], hourly["event_count"], color="seagreen", alpha=0.85)
    ax.set_title("DHW Draw Event Frequency by Hour of Day")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Event count")
    ax.set_xticks(np.arange(0, 24, 2, dtype=int))
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, used_encoding = read_csv_with_encoding_fallback(args.input_csv)
    print(f"Using CSV encoding: {used_encoding}")
    if args.time_col not in df.columns:
        raise ValueError(f"Missing time column: {args.time_col}")

    signal_col = select_signal_column(df, args.signal_col)

    df[args.time_col] = pd.to_datetime(df[args.time_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[args.time_col]).sort_values(args.time_col).reset_index(drop=True)

    events, used_signal, used_threshold = build_events(
        df=df,
        time_col=args.time_col,
        signal_col=signal_col,
        threshold=args.threshold,
    )

    unit = infer_unit(used_signal)

    monthly_size_dist = build_monthly_size_distribution(events, bins=args.bins)
    monthly_freq = build_monthly_frequency(events)
    hourly_dist = build_hourly_distribution(events)

    events_out = args.output_dir / "dhw_draw_events.csv"
    monthly_size_out = args.output_dir / "dhw_event_size_distribution_by_month.csv"
    monthly_freq_out = args.output_dir / "dhw_event_frequency_by_month.csv"
    hourly_out = args.output_dir / "dhw_event_frequency_by_hour.csv"

    events.to_csv(events_out, index=False)
    monthly_size_dist.to_csv(monthly_size_out, index=False)
    monthly_freq.to_csv(monthly_freq_out, index=False)
    hourly_dist.to_csv(hourly_out, index=False)

    plot_monthly_size_histograms(
        events,
        args.output_dir / "dhw_event_size_histograms_by_month.png",
        unit=unit,
        bins=args.bins,
    )
    plot_monthly_frequency(
        monthly_freq,
        args.output_dir / "dhw_event_frequency_by_month.png",
    )
    plot_hourly_distribution(
        hourly_dist,
        args.output_dir / "dhw_event_frequency_by_hour.png",
    )

    print("DHW demand profile extraction complete")
    print(f"Input CSV: {args.input_csv}")
    print(f"Signal column: {used_signal}")
    print(f"Threshold: {used_threshold:g} {unit}")
    print(f"Events extracted: {len(events)}")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
