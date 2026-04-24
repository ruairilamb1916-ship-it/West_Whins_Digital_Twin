#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_OUTPUT_DIR = Path("output/forecast_control_plots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot forecast control strategy comparison figures from three strategy output CSVs."
    )
    parser.add_argument("--normal-csv", type=Path, required=True, help="Forecast output CSV for normal strategy")
    parser.add_argument(
        "--solar-priority-csv",
        type=Path,
        required=True,
        help="Forecast output CSV for solar_priority strategy",
    )
    parser.add_argument("--preheat-csv", type=Path, required=True, help="Forecast output CSV for preheat strategy")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save plots")
    return parser.parse_args()


def _read_strategy_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"CSV missing required 'timestamp' column: {path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    return df.sort_values("timestamp").reset_index(drop=True)


def _find_first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _plot_series_comparison(
    data: dict[str, pd.DataFrame],
    y_col_candidates: list[str],
    ylabel: str,
    title: str,
    output_path: Path,
) -> bool:
    plt.figure(figsize=(12, 5))

    plotted_any = False
    for strategy, df in data.items():
        col = _find_first_column(df, y_col_candidates)
        if col is None:
            continue
        y = pd.to_numeric(df[col], errors="coerce")
        plt.plot(df["timestamp"], y, label=strategy, linewidth=1.4)
        plotted_any = True

    if not plotted_any:
        plt.close()
        return False

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Timestamp")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def main() -> None:
    args = parse_args()

    strategy_data = {
        "normal": _read_strategy_csv(args.normal_csv),
        "solar_priority": _read_strategy_csv(args.solar_priority_csv),
        "preheat": _read_strategy_csv(args.preheat_csv),
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    top_temp_path = output_dir / "top_tank_temperature_comparison.png"
    if _plot_series_comparison(
        data=strategy_data,
        y_col_candidates=["tank_top_pred_c", "tank_top_sim_c", "tank_top_c"],
        ylabel="Top tank temperature [C]",
        title="Forecast Control Comparison: Top Tank Temperature",
        output_path=top_temp_path,
    ):
        saved_paths.append(top_temp_path)

    ashp_heat_path = output_dir / "predicted_ashp_heat_comparison.png"
    if _plot_series_comparison(
        data=strategy_data,
        y_col_candidates=["predicted_ashp_heat_kwh", "Q_ashp_kwh", "ashp_heat_kwh"],
        ylabel="Predicted ASHP heat [kWh per timestep]",
        title="Forecast Control Comparison: Predicted ASHP Heat",
        output_path=ashp_heat_path,
    ):
        saved_paths.append(ashp_heat_path)

    dhw_energy_path = output_dir / "predicted_dhw_draw_energy_comparison.png"
    if _plot_series_comparison(
        data=strategy_data,
        y_col_candidates=["predicted_dhw_draw_energy_kwh", "dhw_draw_energy_kwh"],
        ylabel="Predicted DHW draw energy [kWh per timestep]",
        title="Forecast Control Comparison: Predicted DHW Draw Energy",
        output_path=dhw_energy_path,
    ):
        saved_paths.append(dhw_energy_path)

    solar_path = output_dir / "solar_input_comparison.png"
    solar_available = _plot_series_comparison(
        data=strategy_data,
        y_col_candidates=["st_kwh", "solar_kwh", "st_energy_kwh", "st_power_kw", "solar_power_kw", "ST Power [kW]"],
        ylabel="Solar input [kWh or kW per timestep]",
        title="Forecast Control Comparison: Solar Input (if available)",
        output_path=solar_path,
    )
    if solar_available:
        saved_paths.append(solar_path)

    print("Forecast control comparison plotting complete")
    print(f"Output directory: {output_dir}")
    for p in saved_paths:
        print(f"Saved: {p}")
    if not solar_available:
        print("Solar input plot skipped: no solar column found in provided strategy CSVs")


if __name__ == "__main__":
    main()
