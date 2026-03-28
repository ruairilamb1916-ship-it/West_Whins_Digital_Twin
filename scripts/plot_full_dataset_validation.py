#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pick_column(columns, candidates, label):
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(
        f"Could not find {label} column. Tried: {candidates}\n"
        f"Available columns: {list(columns)}"
    )


def main() -> None:
    csv_path = Path("output/ashp_heat_per_timestep.csv")
    outdir = Path("output/plots")
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")

    meas_col = pick_column(
        df.columns,
        ["Q_meas_backcalc_kwh", "Q_ashp_backcalc_kwh", "ashp_heat_measured"],
        "measured ASHP heat",
    )
    sim_col = pick_column(
        df.columns,
        ["Q_fit_kwh", "Q_fit", "ashp_heat_simulated"],
        "simulated ASHP heat",
    )

    plot_df = df[["timestamp", meas_col, sim_col]].copy().dropna()
    plot_df = plot_df.set_index("timestamp")

    y_true = plot_df[meas_col]
    y_pred = plot_df[sim_col]
    residual = y_pred - y_true

    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mae = np.mean(np.abs(y_pred - y_true))
    bias = np.mean(y_pred - y_true)

    total_measured = y_true.sum()
    total_simulated = y_pred.sum()
    pct_error = 100.0 * (total_simulated - total_measured) / total_measured if total_measured != 0 else np.nan

    print(f"Measured column : {meas_col}")
    print(f"Simulated column: {sim_col}")
    print(f"Date range      : {plot_df.index.min()} -> {plot_df.index.max()}")
    print(f"N points        : {len(plot_df)}")
    print()
    print(f"RMSE            : {rmse:.3f} kWh/timestep")
    print(f"MAE             : {mae:.3f} kWh/timestep")
    print(f"Bias            : {bias:.3f} kWh/timestep")
    print(f"Total measured  : {total_measured:.1f} kWh")
    print(f"Total simulated : {total_simulated:.1f} kWh")
    print(f"Total error     : {pct_error:.2f} %")

    # 1) Full-resolution plot
    plt.figure(figsize=(15, 5))
    plt.plot(plot_df.index, y_true, label="Measured ASHP heat", linewidth=1.0)
    plt.plot(plot_df.index, y_pred, label="Simulated ASHP heat", linewidth=1.0)
    plt.xlabel("Time")
    plt.ylabel("ASHP heat per timestep (kWh)")
    plt.title("ASHP Validation - Full Dataset")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "ashp_validation_full_resolution.png", dpi=300)
    plt.close()

    # 2) Daily total energy plot
    daily = plot_df.resample("D").sum()

    plt.figure(figsize=(15, 5))
    plt.plot(daily.index, daily[meas_col], label="Measured daily heat")
    plt.plot(daily.index, daily[sim_col], label="Simulated daily heat")
    plt.xlabel("Time")
    plt.ylabel("ASHP heat (kWh/day)")
    plt.title("ASHP Validation - Daily Total Heat")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "ashp_validation_daily_totals.png", dpi=300)
    plt.close()

    # 3) Monthly total energy plot
    monthly = plot_df.resample("M").sum()

    plt.figure(figsize=(15, 5))
    plt.plot(monthly.index, monthly[meas_col], marker="o", label="Measured monthly heat")
    plt.plot(monthly.index, monthly[sim_col], marker="o", label="Simulated monthly heat")
    plt.xlabel("Time")
    plt.ylabel("ASHP heat (kWh/month)")
    plt.title("ASHP Validation - Monthly Total Heat")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "ashp_validation_monthly_totals.png", dpi=300)
    plt.close()

    # 4) Residuals over time
    plt.figure(figsize=(15, 4))
    plt.plot(plot_df.index, residual, linewidth=0.8)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Sim - Meas (kWh)")
    plt.title("ASHP Validation - Residuals")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "ashp_validation_residuals.png", dpi=300)
    plt.close()

    # 5) Parity plot
    xy_max = max(y_true.max(), y_pred.max())
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.25, s=10)
    plt.plot([0, xy_max], [0, xy_max], linestyle="--")
    plt.xlabel("Measured ASHP heat (kWh)")
    plt.ylabel("Simulated ASHP heat (kWh)")
    plt.title("ASHP Validation - Parity Plot")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "ashp_validation_parity.png", dpi=300)
    plt.close()

    # Save metrics to CSV
    metrics = pd.DataFrame(
        {
            "metric": ["rmse_kwh_per_timestep", "mae_kwh_per_timestep", "bias_kwh_per_timestep",
                       "total_measured_kwh", "total_simulated_kwh", "total_error_percent"],
            "value": [rmse, mae, bias, total_measured, total_simulated, pct_error],
        }
    )
    metrics.to_csv(outdir / "ashp_validation_metrics.csv", index=False)

    print()
    print("Saved:")
    for name in [
        "ashp_validation_full_resolution.png",
        "ashp_validation_daily_totals.png",
        "ashp_validation_monthly_totals.png",
        "ashp_validation_residuals.png",
        "ashp_validation_parity.png",
        "ashp_validation_metrics.csv",
    ]:
        print(outdir / name)


if __name__ == "__main__":
    main()
