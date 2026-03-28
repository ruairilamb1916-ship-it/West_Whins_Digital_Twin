#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PARAMS_JSON = Path("output/params.json")
DEFAULT_OUTPUT_CSV = Path("output/forecast_control_strategy_comparison.csv")
STRATEGIES = ["normal", "solar_priority", "preheat"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare simple forecast control strategies using run_forecast_simulation.py"
    )
    parser.add_argument("--forecast-csv", type=Path, required=True, help="Forecast input CSV")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_JSON, help="Trained params.json")
    parser.add_argument(
        "--comfort-threshold-c",
        type=float,
        default=45.0,
        help="Top-node comfort threshold used for comfort shortfall counting and served-demand proxy.",
    )
    parser.add_argument(
        "--max-below-comfort-steps",
        type=int,
        default=24,
        help="Maximum acceptable timesteps below comfort for a strategy to be considered comfort-acceptable.",
    )
    return parser.parse_args()


def _infer_dt_hours(ts: pd.Series) -> float:
    diffs = ts.sort_values().diff().dropna()
    if diffs.empty:
        return 0.5
    return float(diffs.dt.total_seconds().median() / 3600.0)


def _find_first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _forecast_solar_input_kwh(forecast_df: pd.DataFrame) -> float | None:
    st_kwh_col = _find_first_column(forecast_df, ["st_kwh", "solar_kwh", "st_energy_kwh"])
    if st_kwh_col is not None:
        return float(pd.to_numeric(forecast_df[st_kwh_col], errors="coerce").fillna(0.0).sum())

    st_kw_col = _find_first_column(forecast_df, ["st_power_kw", "solar_power_kw", "ST Power [kW]"])
    if st_kw_col is not None:
        dt_h = _infer_dt_hours(forecast_df["timestamp"])
        st_kw = pd.to_numeric(forecast_df[st_kw_col], errors="coerce").fillna(0.0)
        return float((st_kw * dt_h).sum())

    return None


def _run_one_strategy(
    mode: str,
    forecast_csv: Path,
    params_json: Path,
    temp_output_csv: Path,
) -> pd.DataFrame:
    cmd = [
        sys.executable,
        "scripts/run_forecast_simulation.py",
        "--params",
        str(params_json),
        "--forecast-csv",
        str(forecast_csv),
        "--output",
        str(temp_output_csv),
        "--control-mode",
        mode,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return pd.read_csv(temp_output_csv)


def _compute_metrics(
    strategy_name: str,
    pred_df: pd.DataFrame,
    comfort_threshold_c: float,
    solar_input_kwh: float | None,
) -> dict[str, float | int | str]:
    top_c = pd.to_numeric(pred_df["tank_top_pred_c"], errors="coerce").fillna(np.nan)
    ashp_kwh = pd.to_numeric(pred_df["predicted_ashp_heat_kwh"], errors="coerce").fillna(0.0)
    dhw_kwh = pd.to_numeric(pred_df["predicted_dhw_draw_energy_kwh"], errors="coerce").fillna(0.0)

    below_comfort_mask = top_c < float(comfort_threshold_c)

    # Simple served-demand proxy: demand in timesteps where top node is at/above comfort threshold.
    dhw_served_kwh = float(dhw_kwh[~below_comfort_mask].sum())

    metrics: dict[str, float | int | str] = {
        "strategy": strategy_name,
        "total_predicted_ashp_heat_kwh": float(ashp_kwh.sum()),
        "total_predicted_dhw_demand_served_kwh": dhw_served_kwh,
        "min_top_tank_temp_c": float(top_c.min()),
        "timesteps_top_below_comfort": int(below_comfort_mask.sum()),
    }

    if solar_input_kwh is not None:
        total_charge = float(ashp_kwh.sum()) + float(solar_input_kwh)
        solar_fraction = float(solar_input_kwh / total_charge) if total_charge > 0.0 else np.nan
        metrics["solar_thermal_input_kwh"] = float(solar_input_kwh)
        metrics["solar_charge_fraction_proxy"] = solar_fraction

    return metrics


def _choose_best_strategy(summary_df: pd.DataFrame, max_below_comfort_steps: int) -> tuple[str, str, str]:
    has_solar_fraction = "solar_charge_fraction_proxy" in summary_df.columns

    acceptable = summary_df.loc[
        summary_df["timesteps_top_below_comfort"] <= int(max_below_comfort_steps)
    ].copy()

    if has_solar_fraction:
        acceptable["solar_charge_fraction_proxy"] = pd.to_numeric(
            acceptable["solar_charge_fraction_proxy"], errors="coerce"
        )
        summary_with_solar = summary_df.copy()
        summary_with_solar["solar_charge_fraction_proxy"] = pd.to_numeric(
            summary_with_solar["solar_charge_fraction_proxy"], errors="coerce"
        )
    else:
        summary_with_solar = summary_df

    if not acceptable.empty:
        sort_cols = ["total_predicted_ashp_heat_kwh"]
        ascending = [True]
        if has_solar_fraction:
            sort_cols.append("solar_charge_fraction_proxy")
            ascending.append(False)
        # Stable tie-breakers.
        sort_cols.extend(["timesteps_top_below_comfort", "min_top_tank_temp_c"])
        ascending.extend([True, False])

        ranked = acceptable.sort_values(by=sort_cols, ascending=ascending).reset_index(drop=True)
        best = str(ranked.loc[0, "strategy"])
        criteria = (
            "Ranking criteria: comfort gate first "
            f"(timesteps_top_below_comfort <= {int(max_below_comfort_steps)}), then minimize "
            "total_predicted_ashp_heat_kwh, then maximize solar_charge_fraction_proxy."
            if has_solar_fraction
            else (
                "Ranking criteria: comfort gate first "
                f"(timesteps_top_below_comfort <= {int(max_below_comfort_steps)}), then minimize "
                "total_predicted_ashp_heat_kwh."
            )
        )
        reason = "Selected from comfort-acceptable strategies using energy-first ranking"
        return best, criteria, reason

    # Fallback if none meet comfort gate: keep comfort as priority, then energy and solar.
    fallback_cols = ["timesteps_top_below_comfort", "total_predicted_ashp_heat_kwh"]
    fallback_asc = [True, True]
    if has_solar_fraction:
        fallback_cols.append("solar_charge_fraction_proxy")
        fallback_asc.append(False)
    fallback_cols.append("min_top_tank_temp_c")
    fallback_asc.append(False)

    ranked = summary_with_solar.sort_values(by=fallback_cols, ascending=fallback_asc).reset_index(drop=True)
    best = str(ranked.loc[0, "strategy"])
    criteria = (
        "Ranking criteria: comfort gate first "
        f"(timesteps_top_below_comfort <= {int(max_below_comfort_steps)}), then minimize "
        "total_predicted_ashp_heat_kwh, then maximize solar_charge_fraction_proxy."
        if has_solar_fraction
        else (
            "Ranking criteria: comfort gate first "
            f"(timesteps_top_below_comfort <= {int(max_below_comfort_steps)}), then minimize "
            "total_predicted_ashp_heat_kwh."
        )
    )
    reason = "No strategy met comfort gate; selected best fallback with fewest comfort violations"
    return best, criteria, reason


def main() -> None:
    args = parse_args()
    output_csv = DEFAULT_OUTPUT_CSV

    fc = pd.read_csv(args.forecast_csv)
    if "timestamp" not in fc.columns:
        raise ValueError("Forecast CSV must include 'timestamp'")
    fc["timestamp"] = pd.to_datetime(fc["timestamp"], errors="raise")

    solar_input_kwh = _forecast_solar_input_kwh(fc)

    rows: list[dict[str, float | int | str]] = []

    with tempfile.TemporaryDirectory(prefix="forecast_control_compare_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        for mode in STRATEGIES:
            out_csv = tmp_dir_path / f"forecast_{mode}.csv"
            pred = _run_one_strategy(
                mode=mode,
                forecast_csv=args.forecast_csv,
                params_json=args.params,
                temp_output_csv=out_csv,
            )
            rows.append(
                _compute_metrics(
                    strategy_name=mode,
                    pred_df=pred,
                    comfort_threshold_c=float(args.comfort_threshold_c),
                    solar_input_kwh=solar_input_kwh,
                )
            )

    summary = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)

    best, criteria_text, selection_reason = _choose_best_strategy(
        summary,
        max_below_comfort_steps=int(args.max_below_comfort_steps),
    )
    best_row = summary.loc[summary["strategy"] == best].iloc[0]

    print("Forecast control strategy comparison complete")
    print(f"Forecast CSV: {args.forecast_csv}")
    print(f"Params JSON: {args.params}")
    print(f"Comparison CSV: {output_csv.resolve()}")
    print("Summary metrics by strategy:")
    cols = [
        "strategy",
        "total_predicted_ashp_heat_kwh",
        "total_predicted_dhw_demand_served_kwh",
        "min_top_tank_temp_c",
        "timesteps_top_below_comfort",
    ]
    optional_cols = ["solar_thermal_input_kwh", "solar_charge_fraction_proxy"]
    for c in optional_cols:
        if c in summary.columns:
            cols.append(c)
    print(summary[cols].to_string(index=False))
    print(criteria_text)

    print(
        "Best strategy: "
        f"{best} "
        f"(below-comfort steps={int(best_row['timesteps_top_below_comfort'])}, "
        f"min top={float(best_row['min_top_tank_temp_c']):.2f} C, "
        f"ASHP heat={float(best_row['total_predicted_ashp_heat_kwh']):.2f} kWh)"
    )
    if "solar_charge_fraction_proxy" in best_row.index:
        print(
            "Selection explanation: "
            f"{selection_reason}; solar fraction={float(best_row['solar_charge_fraction_proxy']):.4f}."
        )
    else:
        print(f"Selection explanation: {selection_reason}.")


if __name__ == "__main__":
    main()
