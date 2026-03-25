#!/usr/bin/env python3
"""
run_stage1.py – Stage-1 Digital Twin Pipeline
==============================================

Orchestrates data loading, parameter identification, and evaluation
for the West Whins DHW system grey-box model.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import data_loader, identification, ashp_model, tank_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent


def _resolve_fit_profile(fit_profile: str, max_nfev: int) -> dict:
    """Return effective run settings for a named fit profile.

    `full` preserves the existing behaviour.
    `fast` is the ASHP-label iteration path: seasonal sampling plus no tank refit.
    `fast-refit` keeps a cheap tank refit for a closer approximation.
    """
    profile = fit_profile.lower()
    if profile == "full":
        return {
            "fit_profile": profile,
            "max_nfev": max_nfev,
            "default_n_weeks": None,
            "default_sample_blocks": None,
            "fit_tank": True,
            "tank_fit_kwargs": {},
        }
    if profile == "fast":
        return {
            "fit_profile": profile,
            "max_nfev": min(max_nfev, 40),
            "default_n_weeks": 2,
            "default_sample_blocks": 4,
            "fit_tank": False,
            "tank_fit_kwargs": {},
        }
    if profile == "fast-refit":
        return {
            "fit_profile": profile,
            "max_nfev": min(max_nfev, 40),
            "default_n_weeks": 2,
            "default_sample_blocks": 4,
            "fit_tank": True,
            "tank_fit_kwargs": {
                "rollout_weight": 0.05,
                "rollout_horizon": 12,
                "rollout_stride": 336,
            },
        }
    raise ValueError(f"Unknown fit_profile={fit_profile!r}")


def main(
    csv_path: Path | None = None,
    yaml_path: Path | None = None,
    output_dir: Path | None = None,
    train_frac: float = 0.7,
    max_nfev: int = 300,
    n_weeks: int | None = None,
    start_week: int = 0,
    sample_blocks: int | None = None,
    fit_profile: str = "full",
) -> dict:
    csv_path = csv_path or ROOT / "data" / "FullDS_Findhorn.csv"
    yaml_path = yaml_path or ROOT / "column_mapping.yaml"
    output_dir = output_dir or ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data from %s", csv_path)
    df = data_loader.load_and_clean(csv_path, yaml_path)
    logger.info("Columns in df: %s", df.columns.tolist())

    profile = _resolve_fit_profile(fit_profile, max_nfev)
    max_nfev = profile["max_nfev"]
    if n_weeks is None:
        n_weeks = profile["default_n_weeks"]
    if sample_blocks is None:
        sample_blocks = profile["default_sample_blocks"]
    logger.info(
        "Using fit profile '%s' (max_nfev=%d, fit_tank=%s, n_weeks=%s, sample_blocks=%s)",
        profile["fit_profile"],
        max_nfev,
        profile["fit_tank"],
        n_weeks,
        sample_blocks,
    )

    tank_cols = ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]
    df = df.dropna(subset=tank_cols, how="all")
    logger.info("After dropping all-NaN tank rows: %d rows", len(df))

    STEP = 7 * 48  # rows per week (48 half-hour steps/day)

    if sample_blocks is not None:
        # Pick `sample_blocks` evenly-spaced contiguous blocks of `n_weeks` weeks
        # (default 3 weeks per block) spread across the full dataset.
        # Blocks are contiguous so dt=30 min is preserved within each block.
        block_weeks = n_weeks if n_weeks is not None else 3
        block_size = block_weeks * STEP
        total_rows = len(df)
        if sample_blocks * block_size > total_rows:
            logger.warning(
                "sample_blocks=%d × block=%d rows exceeds dataset (%d); using all data.",
                sample_blocks, block_size, total_rows,
            )
            sample_blocks = 1
            block_size = total_rows
        starts = np.linspace(0, total_rows - block_size, sample_blocks, dtype=int)
        blocks = [df.iloc[int(s): int(s) + block_size] for s in starts]
        df = pd.concat(blocks)
        logger.info(
            "Sampled %d blocks × %d weeks = %d rows (starts at weeks %s)",
            sample_blocks, block_weeks, len(df),
            [int(s) // STEP for s in starts],
        )
    elif n_weeks is not None:
        # 48 half-hour steps per day × 7 days per week
        start_row = start_week * STEP
        n_rows = n_weeks * STEP
        df = df.iloc[start_row: start_row + n_rows].copy()
        logger.info(
            "Subset: weeks %d–%d (%d rows, starting %s)",
            start_week, start_week + n_weeks, len(df), df.index[0],
        )
    elif start_week > 0:
        df = df.iloc[start_week * STEP:].copy()
        logger.info("Starting from week %d (%s, %d rows)", start_week, df.index[0], len(df))

    ordering = data_loader.node_ordering_check(df)
    logger.info("Node ordering satisfied: %.1f %%", ordering.mean() * 100)

    logger.info("Running identification (train_frac=%.2f) …", train_frac)
    id_result, df_train, df_val = identification.run_identification(
        df,
        train_frac=train_frac,
        max_nfev=max_nfev,
        fit_tank=profile["fit_tank"],
        tank_fit_kwargs=profile["tank_fit_kwargs"],
    )

    params_file = output_dir / "params.json"
    _save_params(id_result, params_file)
    logger.info("Parameters saved to %s", params_file)

    # ------------------------------------------------------------------
    # Validation diagnostics (teacher-forced vs closed-loop)
    # ------------------------------------------------------------------
    val_inputs = identification.prepare_inputs(df_val, id_result.ashp_params)
    T_val = val_inputs["T_meas"]

    def _rmse(a: np.ndarray, b: np.ndarray) -> float:
        return np.sqrt(np.nanmean((a - b) ** 2))

    # One-step-ahead (teacher forcing) uses measured T[k] to predict T[k+1].
    T_tf = np.zeros_like(T_val)
    T_tf[0] = T_val[0]
    for k in range(len(T_val) - 1):
        T_tf[k + 1] = tank_model.tank_step(
            T_val[k],
            float(val_inputs["Q_st"][k]), float(val_inputs["Q_ashp"][k]),
            float(val_inputs["Q_imm"][k]), float(val_inputs["T_amb"][k]),
            float(val_inputs["V_draw"][k]), float(val_inputs["T_cold"][k]),
            id_result.tank_params,
        )
    rmse_tf = _rmse(T_tf[1:], T_val[1:])

    # Closed-loop (autonomous) prediction uses the model's own state each step.
    T_cl = identification.simulate_closed_loop(
        T_val[0],
        val_inputs["Q_st"],
        val_inputs["Q_imm"],
        val_inputs["T_amb"],
        val_inputs["V_draw"],
        val_inputs["T_cold"],
        val_inputs["T_out"],
        val_inputs["P_meas"],
        id_result.ashp_params,
        id_result.tank_params,
    )
    rmse_cl = _rmse(T_cl[1:-1], T_val[1:])

    logger.info(
        "Validation RMSE: one-step-ahead = %.3f °C, closed-loop = %.3f °C (drift expected).",
        rmse_tf,
        rmse_cl,
    )

    # ---- Export ASHP heat timeseries CSV ----------------------------------
    timestamp = df.index

    T_sink = ashp_model.sink_proxy(
        df["tank_mid_c"].to_numpy(),
        df["tank_top_c"].to_numpy(),
    )

    split_idx = len(df_train)
    split = np.array(["validation"] * len(df), dtype=object)
    split[:split_idx] = "train"

    df_all = pd.concat([df_train, df_val])

    P_fit_kw = ashp_model.predict_power(
        df["t_out_c"].to_numpy(),
        T_sink,
        id_result.ashp_params,
    )
    Q_fit_kw = ashp_model.predict_capacity(
        df["t_out_c"].to_numpy(),
        T_sink,
        id_result.ashp_params,
    )

    export_df = pd.DataFrame({
        "timestamp": timestamp,
        "split": split,
        "t_out_c": df["t_out_c"].to_numpy(),
        "t_amb_c": df["t_amb_c"].to_numpy(),
        "tank_mid_c": df["tank_mid_c"].to_numpy(),
        "tank_top_c": df["tank_top_c"].to_numpy(),
        "T_sink_c": T_sink,
        "ashp_inst_kwh": df["ashp_inst_kwh"].to_numpy(),
        "Q_meas_backcalc_kwh": df_all["Q_ashp_backcalc_kwh"].to_numpy(),
        "P_fit_kw": P_fit_kw,
        "Q_fit_kw": Q_fit_kw,
        "Q_fit_kwh": Q_fit_kw * 0.5,
    })

    out_csv = output_dir / "ashp_heat_per_timestep.csv"
    export_df.to_csv(out_csv, index=False)
    logger.info("Saved ASHP heat timeseries CSV to %s", out_csv)

    return {"status": "identification completed"}


def _save_params(id_result: identification.IdentificationResult, path: Path) -> None:
    data = {
        "tank": {
            "UA_loss": id_result.tank_params.UA_loss.tolist(),
            "UA_adj": id_result.tank_params.UA_adj.tolist(),
            "f_st": id_result.tank_params.f_st.tolist(),
            "f_ashp": id_result.tank_params.f_ashp.tolist(),
            "f_imm": id_result.tank_params.f_imm.tolist(),
            "mix_coeff": id_result.tank_params.mix_coeff,
            "alpha_draw": id_result.tank_params.alpha_draw,
            "T_mains": id_result.tank_params.T_mains,
        },
        "ashp": {
            "a": id_result.ashp_params.a.tolist(),
            "b": id_result.ashp_params.b.tolist(),
        },
        "hx_effectiveness": id_result.hx_effectiveness,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage-1 DHW Digital Twin")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--yaml", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--weeks", type=int, default=None,
                        help="Number of contiguous weeks to fit on (combine with --start-week).")
    parser.add_argument("--start-week", type=int, default=0,
                        help="Week offset into the dataset (0=Dec 2023). Use to pick a season.")
    parser.add_argument("--sample-blocks", type=int, default=None,
                        help="Pick N evenly-spaced blocks of --weeks weeks across the full dataset "
                             "for seasonal coverage. E.g. --sample-blocks 4 --weeks 3 = 4×3 week "
                             "blocks (winter/spring/summer/autumn).")
    parser.add_argument("--fit-profile", choices=["full", "fast", "fast-refit"], default="full",
                        help="Run profile. 'fast' is for quick ASHP-label iteration and skips tank re-fit; "
                             "'fast-refit' keeps a cheap tank fit on sampled seasonal blocks.")
    args = parser.parse_args()
    main(args.csv, args.yaml, args.output, args.train_frac, args.max_nfev,
         args.weeks, args.start_week, args.sample_blocks, args.fit_profile)