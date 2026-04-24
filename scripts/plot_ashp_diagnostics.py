#!/usr/bin/env python3
"""Create simple diagnostic plots from output/debug_ashp_map.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path)


def _has_target(df: pd.DataFrame) -> bool:
    return "COP_target" in df.columns and df["COP_target"].notna().any()


def plot_cop_fit_vs_target(df: pd.DataFrame, outdir: Path) -> None:
    if not _has_target(df):
        return

    data = df[["COP_fit_plot", "COP_target_plot"]].dropna()
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(data["COP_target_plot"], data["COP_fit_plot"], s=16, alpha=0.6)

    ref_min = min(data["COP_fit_plot"].min(), data["COP_target_plot"].min())
    ref_max = max(data["COP_fit_plot"].max(), data["COP_target_plot"].max())
    ax.plot([ref_min, ref_max], [ref_min, ref_max], "k--", linewidth=1)

    ax.set_xlabel("COP_target")
    ax.set_ylabel("COP_fit")
    ax.set_xlim(1.0, 5.0)
    ax.set_ylim(1.0, 5.0)
    ax.set_title("ASHP COP Fit vs Target (Clipped for Visualisation)")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, outdir / "cop_fit_vs_target.png")


def plot_cop_vs_tout(df: pd.DataFrame, outdir: Path) -> None:
    data_fit = df[["t_out_c", "COP_fit_plot"]].dropna()
    if data_fit.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(data_fit["t_out_c"], data_fit["COP_fit_plot"], s=14, alpha=0.5, label="COP_fit")

    if _has_target(df):
        data_target = df[["t_out_c", "COP_target_plot"]].dropna()
        if not data_target.empty:
            ax.scatter(
                data_target["t_out_c"],
                data_target["COP_target_plot"],
                s=14,
                alpha=0.5,
                label="COP_target",
            )

    ax.set_xlabel("t_out_c")
    ax.set_ylabel("COP")
    ax.set_title("COP vs Outdoor Temperature")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_fig(fig, outdir / "cop_vs_tout.png")


def plot_cop_hist(df: pd.DataFrame, outdir: Path) -> None:
    fit = df["COP_fit_plot"].dropna()
    if fit.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(fit, bins=30, alpha=0.6, label="COP_fit", density=True)

    if _has_target(df):
        target = df["COP_target_plot"].dropna()
        if not target.empty:
            ax.hist(target, bins=30, alpha=0.6, label="COP_target", density=True)

    ax.set_xlabel("COP")
    ax.set_ylabel("Density")
    ax.set_xlim(1.0, 5.0)
    ax.set_title("COP Distribution Comparison (Clipped for Visualisation)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_fig(fig, outdir / "cop_hist.png")


def plot_cop_residuals(df: pd.DataFrame, outdir: Path) -> None:
    if not _has_target(df):
        return

    data = df[["COP_fit_plot", "COP_target_plot"]].dropna()
    if data.empty:
        return

    residual = data["COP_fit_plot"] - data["COP_target_plot"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(data["COP_target_plot"], residual, s=16, alpha=0.6)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("COP_target")
    ax.set_ylabel("COP_fit - COP_target")
    ax.set_xlim(1.0, 5.0)
    ax.set_title("COP Residuals vs Target")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, outdir / "cop_residuals.png")


def plot_qfit_vs_pfit(df: pd.DataFrame, outdir: Path) -> None:
    data = df[["P_fit", "Q_fit"]].dropna()
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(data["P_fit"], data["Q_fit"], s=16, alpha=0.6)
    ax.set_xlabel("P_fit")
    ax.set_ylabel("Q_fit")
    ax.set_title("Q_fit vs P_fit")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, outdir / "qfit_vs_pfit.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ASHP diagnostics from debug_ashp_map.csv")
    parser.add_argument("--csv", type=Path, default=Path("output/debug_ashp_map.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("output/plots"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["COP_fit_plot"] = df["COP_fit"].clip(1.0, 5.0)
    if "COP_target" in df.columns:
        df["COP_target_plot"] = df["COP_target"].clip(1.0, 5.0)
    args.outdir.mkdir(parents=True, exist_ok=True)

    plot_cop_fit_vs_target(df, args.outdir)
    plot_cop_vs_tout(df, args.outdir)
    plot_cop_hist(df, args.outdir)
    plot_cop_residuals(df, args.outdir)
    plot_qfit_vs_pfit(df, args.outdir)


if __name__ == "__main__":
    main()