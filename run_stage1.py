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
DEFAULT_DHW_PATH = ROOT / "output" / "dhw_stochastic_model" / "dhw_tank_input_2024_scaled_2000.csv"
DHW_SMOOTHING_WINDOW = 3


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
            "default_train_frac": None,
            "fit_tank": True,
            "tank_fit_kwargs": {
                "rollout_weight": 0.05,
                "rollout_horizon": 12,
                "rollout_stride": 336,
            },
        }
    if profile == "fast_ashp":
        return {
            "fit_profile": profile,
            "max_nfev": min(max_nfev, 20),
            "default_n_weeks": 12,
            "default_sample_blocks": None,
            "default_train_frac": 0.5,
            "fit_tank": False,
            "tank_fit_kwargs": {},
        }
    raise ValueError(f"Unknown fit_profile={fit_profile!r}")


def _load_tank_params_if_available(path: Path) -> tank_model.TankParams | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
        tank = payload.get("tank", {})
        return tank_model.TankParams(
            UA_loss=np.asarray(tank["UA_loss"], dtype=float),
            UA_adj=np.asarray(tank["UA_adj"], dtype=float),
            f_st=np.asarray(tank["f_st"], dtype=float),
            f_ashp=np.asarray(tank["f_ashp"], dtype=float),
            f_imm=np.asarray(tank["f_imm"], dtype=float),
            mix_coeff=float(tank["mix_coeff"]),
            alpha_draw=float(tank["alpha_draw"]),
            T_mains=float(tank["T_mains"]),
        )
    except Exception as exc:
        logger.warning("Failed loading tank params from %s: %s", path, exc)
        return None


def _safe_float(x: float | np.floating | None) -> float:
    if x is None or not np.isfinite(x):
        return float("nan")
    return float(x)


def _merge_dhw_profile(df: pd.DataFrame, dhw_path: Path | None) -> pd.DataFrame:
    merged = df.copy()

    if dhw_path is None:
        dhw_path = DEFAULT_DHW_PATH
    if not dhw_path.exists():
        merged["dhw_draw_energy_kwh"] = 0.0
        logger.info("No DHW profile found at %s; assuming zero DHW withdrawal.", dhw_path)
        return merged

    dhw_df = pd.read_csv(dhw_path)
    if "timestamp" not in dhw_df.columns:
        raise ValueError(f"Missing timestamp column in DHW profile: {dhw_path}")

    dhw_df["timestamp"] = pd.to_datetime(dhw_df["timestamp"], errors="raise")
    if "dhw_draw_size_l" not in dhw_df.columns:
        dhw_df["dhw_draw_size_l"] = 0.0
    if "dhw_draw_energy_kwh" not in dhw_df.columns:
        dhw_df["dhw_draw_energy_kwh"] = 0.0
    dhw_df["dhw_draw_size_l"] = pd.to_numeric(
        dhw_df["dhw_draw_size_l"], errors="coerce"
    ).fillna(0.0)
    dhw_df["dhw_draw_energy_kwh"] = pd.to_numeric(
        dhw_df["dhw_draw_energy_kwh"], errors="coerce"
    ).fillna(0.0)

    merged = (
        merged.reset_index()
        .merge(
            dhw_df[["timestamp", "dhw_draw_size_l", "dhw_draw_energy_kwh"]],
            left_on="time",
            right_on="timestamp",
            how="left",
        )
        .drop(columns=["timestamp"])
        .set_index("time")
        .sort_index()
    )
    if "dhw_draw_size_l" not in merged.columns:
        merged["dhw_draw_size_l"] = 0.0
    if "dhw_draw_energy_kwh" not in merged.columns:
        merged["dhw_draw_energy_kwh"] = 0.0
    matched_rows = int(merged["dhw_draw_energy_kwh"].notna().sum())
    merged["dhw_draw_size_l"] = merged["dhw_draw_size_l"].fillna(0.0)
    merged["dhw_draw_energy_kwh"] = merged["dhw_draw_energy_kwh"].fillna(0.0)

    original_total_l = float(merged["dhw_draw_size_l"].sum())
    original_total_dhw = float(merged["dhw_draw_energy_kwh"].sum())
    if original_total_l > 0:
        smoothed_l = (
            merged["dhw_draw_size_l"]
            .rolling(window=DHW_SMOOTHING_WINDOW, min_periods=1, center=True)
            .mean()
            .fillna(0.0)
        )
        smoothed_total_l = float(smoothed_l.sum())
        if smoothed_total_l > 0:
            smoothed_l *= original_total_l / smoothed_total_l
        merged["dhw_draw_size_l"] = smoothed_l

    if original_total_dhw > 0:
        smoothed_kwh = (
            merged["dhw_draw_energy_kwh"]
            .rolling(window=DHW_SMOOTHING_WINDOW, min_periods=1, center=True)
            .mean()
            .fillna(0.0)
        )
        smoothed_total_dhw = float(smoothed_kwh.sum())
        if smoothed_total_dhw > 0:
            smoothed_kwh *= original_total_dhw / smoothed_total_dhw
        merged["dhw_draw_energy_kwh"] = smoothed_kwh

    logger.info(
        "Merged DHW profile from %s: matched %d/%d rows, DHW energy %.2f -> %.2f kWh after %d-step smoothing.",
        dhw_path,
        matched_rows,
        len(merged),
        original_total_dhw,
        float(merged["dhw_draw_energy_kwh"].sum()),
        DHW_SMOOTHING_WINDOW,
    )
    logger.info(
        "Synthetic DHW volume %.2f -> %.2f L after smoothing; withdrawal will be applied via tank draw displacement.",
        original_total_l,
        float(merged["dhw_draw_size_l"].sum()),
    )
    return merged


def _split_summary(
    df_split: pd.DataFrame,
    id_result: identification.IdentificationResult,
    label: str,
    split_inputs: dict | None = None,
    T_pred: np.ndarray | None = None,
) -> dict:
    tank_cols = ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]
    T_meas = df_split[tank_cols].to_numpy(dtype=float)

    if split_inputs is None:
        split_inputs = identification.prepare_inputs(df_split, id_result.ashp_params)
    if T_pred is None:
        T_cl = identification.simulate_closed_loop(
            T_meas[0],
            split_inputs["Q_st"],
            split_inputs["Q_imm"],
            split_inputs["T_amb"],
            split_inputs["V_draw"],
            split_inputs["T_cold"],
            split_inputs["T_out"],
            split_inputs["P_meas"],
            id_result.ashp_params,
            id_result.tank_params,
        )
        T_pred = T_cl[1:]
    T_obs = T_meas[1:]
    n = min(len(T_obs), len(T_pred))
    T_obs = T_obs[:n]
    T_pred = T_pred[:n]

    node_rmse = {
        "T_bottom": _safe_float(np.sqrt(np.nanmean((T_pred[:, 0] - T_obs[:, 0]) ** 2))),
        "T_mid": _safe_float(np.sqrt(np.nanmean((T_pred[:, 1] - T_obs[:, 1]) ** 2))),
        "T_mid_hi": _safe_float(np.sqrt(np.nanmean((T_pred[:, 2] - T_obs[:, 2]) ** 2))),
        "T_top": _safe_float(np.sqrt(np.nanmean((T_pred[:, 3] - T_obs[:, 3]) ** 2))),
    }

    T_sink = ashp_model.sink_proxy(
        df_split["tank_bottom_c"].to_numpy(dtype=float),
        df_split["tank_mid_c"].to_numpy(dtype=float),
    )
    t_out = df_split["t_out_c"].to_numpy(dtype=float)
    p_meas = df_split["ashp_inst_kwh"].fillna(0).to_numpy(dtype=float)
    q_fit_kwh = ashp_model.predict_capacity(t_out, T_sink, id_result.ashp_params) * 0.5
    cop_fit = ashp_model.predict_cop(t_out, T_sink, id_result.ashp_params)

    q_back = (
        df_split["Q_ashp_backcalc_kwh"].to_numpy(dtype=float)
        if "Q_ashp_backcalc_kwh" in df_split.columns
        else np.full(len(df_split), np.nan, dtype=float)
    )

    cop_mask = np.isfinite(q_back) & (p_meas > 0.05) & np.isfinite(cop_fit)
    if np.any(cop_mask):
        cop_true = q_back[cop_mask] / np.maximum(p_meas[cop_mask], 1e-9)
        ape = 100.0 * np.abs(cop_fit[cop_mask] - cop_true) / np.maximum(np.abs(cop_true), 1e-9)
        cop_errors = {
            "median_ape": _safe_float(np.nanmedian(ape)),
            "mean_ape": _safe_float(np.nanmean(ape)),
            "rmse": _safe_float(np.sqrt(np.nanmean((cop_fit[cop_mask] - cop_true) ** 2))),
            "n_samples": int(cop_mask.sum()),
        }
    else:
        cop_errors = {
            "median_ape": float("nan"),
            "mean_ape": float("nan"),
            "rmse": float("nan"),
            "n_samples": 0,
        }

    on_mask = np.isfinite(p_meas) & (p_meas > 0.05) & np.isfinite(cop_fit)
    if np.any(on_mask):
        p_on = p_meas[on_mask]
        q_on = q_fit_kwh[on_mask]
        ashp_kpis = {
            "spf": _safe_float(np.sum(q_on) / max(np.sum(p_on), 1e-9)),
            "mean_cop_on": _safe_float(np.nanmean(cop_fit[on_mask])),
            "frac_cop_above_3": _safe_float(np.mean(cop_fit[on_mask] > 3.0)),
            "ashp_runtime_frac": _safe_float(np.mean(p_meas > 0.05)),
        }
    else:
        ashp_kpis = {
            "spf": float("nan"),
            "mean_cop_on": float("nan"),
            "frac_cop_above_3": float("nan"),
            "ashp_runtime_frac": float("nan"),
        }

    eb_mask = np.isfinite(q_back) & np.isfinite(q_fit_kwh)
    energy_resid = _safe_float(np.nansum(q_fit_kwh[eb_mask] - q_back[eb_mask])) if np.any(eb_mask) else float("nan")

    return {
        "label": label,
        "node_rmse": node_rmse,
        "cop_errors": cop_errors,
        "ashp_kpis": ashp_kpis,
        "ordering_rate": _safe_float(data_loader.node_ordering_check(df_split).mean()),
        "energy_balance_residual_kwh": energy_resid,
        "n_samples": int(len(df_split)),
    }


def main(
    csv_path: Path | None = None,
    yaml_path: Path | None = None,
    output_dir: Path | None = None,
    dhw_path: Path | None = None,
    train_frac: float = 0.7,
    max_nfev: int = 300,
    n_weeks: int | None = None,
    start_week: int = 0,
    sample_blocks: int | None = None,
    fit_profile: str = "full",
) -> dict:
    csv_path = csv_path or ROOT / "data" / "FullDS_Findhorn_clean.csv"
    yaml_path = yaml_path or ROOT / "column_mapping.yaml"
    output_dir = output_dir or ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data from %s", csv_path)
    df = data_loader.load_and_clean(csv_path, yaml_path)
    df = _merge_dhw_profile(df, dhw_path)
    logger.info("Columns in df: %s", df.columns.tolist())

    profile = _resolve_fit_profile(fit_profile, max_nfev)
    max_nfev = profile["max_nfev"]
    if profile.get("default_train_frac") is not None:
        train_frac = profile["default_train_frac"]
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
    if profile["fit_profile"] == "fast_ashp":
        logger.info("fast_ashp seasonal subset selection via --start-week=%d", start_week)

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
    logger.info("Total DHW energy in simulation dataframe: %.2f kWh", float(df["dhw_draw_energy_kwh"].sum()))

    logger.info("Running identification (train_frac=%.2f) …", train_frac)
    fixed_tank_params = None
    summary_fast_path = output_dir / "summary_fast_ashp.json"
    if profile["fit_profile"] == "fast_ashp":
        params_file = output_dir / "params.json"
        fixed_tank_params = _load_tank_params_if_available(params_file)
        if fixed_tank_params is not None:
            logger.info("Loaded existing tank params from %s for fast_ashp profile.", params_file)
        else:
            logger.info("No existing tank params found at %s; using default tank params.", params_file)

    id_result, df_train, df_val = identification.run_identification(
        df,
        train_frac=train_frac,
        max_nfev=max_nfev,
        fit_tank=profile["fit_tank"],
        tank_fit_kwargs=profile["tank_fit_kwargs"],
        fixed_tank_params=fixed_tank_params,
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
            Q_dhw_kwh=float(val_inputs["Q_dhw"][k]),
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
        Q_dhw=val_inputs["Q_dhw"],
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
        df["tank_bottom_c"].to_numpy(),
        df["tank_mid_c"].to_numpy(),
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
    Q_fit_kwh = identification.compute_ashp_heat_kwh(
        df["ashp_inst_kwh"].fillna(0).to_numpy(dtype=float),
        df["t_out_c"].to_numpy(dtype=float),
        T_sink,
        id_result.ashp_params,
        T_bottom=df["tank_bottom_c"].to_numpy(dtype=float),
        T_top=df["tank_top_c"].to_numpy(dtype=float),
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
        "Q_fit_kw": Q_fit_kwh / 0.5,
        "Q_fit_kwh": Q_fit_kwh,
    })

    measured = export_df["Q_meas_backcalc_kwh"].to_numpy(dtype=float)
    simulated = export_df["Q_fit_kwh"].to_numpy(dtype=float)
    scale_mask = np.isfinite(measured) & np.isfinite(simulated)
    if np.any(scale_mask):
        simulated_total = np.sum(simulated[scale_mask])
        if simulated_total > 0:
            scale = np.sum(measured[scale_mask]) / simulated_total
        else:
            scale = 1.0
    else:
        scale = 1.0
    export_df["Q_fit_kwh_scaled"] = export_df["Q_fit_kwh"] * scale

    out_csv = output_dir / "ashp_heat_per_timestep.csv"
    export_df.to_csv(out_csv, index=False)
    logger.info("Saved ASHP heat timeseries CSV to %s", out_csv)

    # ---- Export measured vs simulated tank temperatures (full dataset) ----
    full_inputs = identification.prepare_inputs(df, id_result.ashp_params)
    T_full_cl = identification.simulate_closed_loop(
        full_inputs["T_meas"][0],
        full_inputs["Q_st"],
        full_inputs["Q_imm"],
        full_inputs["T_amb"],
        full_inputs["V_draw"],
        full_inputs["T_cold"],
        full_inputs["T_out"],
        full_inputs["P_meas"],
        id_result.ashp_params,
        id_result.tank_params,
        Q_dhw=full_inputs["Q_dhw"],
    )[1:]

    n_nodes = T_full_cl.shape[1]
    sim_cols = [
        "tank_bottom_sim_c",
        "tank_mid_sim_c",
        "tank_mid_hi_sim_c",
        "tank_top_sim_c",
    ]
    if n_nodes < 4:
        existing_nodes = sim_cols[:n_nodes]
        raise ValueError(
            "Tank simulation returned fewer than 4 nodes; "
            f"available simulated nodes: {existing_nodes}"
        )

    tank_export = {
        "timestamp": df.index,
    }
    measured_map = {
        "tank_bottom_c": "tank_bottom_meas_c",
        "tank_mid_c": "tank_mid_meas_c",
        "tank_mid_hi_c": "tank_mid_hi_meas_c",
        "tank_top_c": "tank_top_meas_c",
    }
    for src_col, out_col in measured_map.items():
        if src_col in df.columns:
            tank_export[out_col] = df[src_col].to_numpy(dtype=float)
    for i, col in enumerate(sim_cols):
        tank_export[col] = T_full_cl[:, i]

    tank_export_df = pd.DataFrame(tank_export)
    tank_csv = output_dir / "tank_temps_measured_vs_sim.csv"
    tank_export_df.to_csv(tank_csv, index=False)
    logger.info("Saved measured vs simulated tank temperatures to %s", tank_csv)

    if profile["fit_profile"] == "fast_ashp":
        train_inputs = identification.prepare_inputs(df_train, id_result.ashp_params)
        T_train_cl = identification.simulate_closed_loop(
            train_inputs["T_meas"][0],
            train_inputs["Q_st"],
            train_inputs["Q_imm"],
            train_inputs["T_amb"],
            train_inputs["V_draw"],
            train_inputs["T_cold"],
            train_inputs["T_out"],
            train_inputs["P_meas"],
            id_result.ashp_params,
            id_result.tank_params,
            Q_dhw=train_inputs["Q_dhw"],
        )
        summary_fast = {
            "train": _split_summary(df_train, id_result, "train", split_inputs=train_inputs, T_pred=T_train_cl[1:]),
            "val": _split_summary(df_val, id_result, "validation", split_inputs=val_inputs, T_pred=T_cl[1:]),
        }
        with open(summary_fast_path, "w") as f:
            json.dump(summary_fast, f, indent=2)
        logger.info("Saved fast ASHP summary to %s", summary_fast_path)

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
            "c": id_result.ashp_params.c.tolist(),
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
    parser.add_argument("--dhw", type=Path, default=None,
                        help="Optional DHW tank-input CSV. Defaults to the standard stochastic DHW output path if present.")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--weeks", type=int, default=None,
                        help="Number of contiguous weeks to fit on (combine with --start-week).")
    parser.add_argument("--start-week", type=int, default=0,
                        help="Week offset into the dataset (0=Dec 2023). Use to pick a season; "
                            "for fast_ashp comparisons, run with e.g. --start-week 0, 12, or 24.")
    parser.add_argument("--sample-blocks", type=int, default=None,
                        help="Pick N evenly-spaced blocks of --weeks weeks across the full dataset "
                             "for seasonal coverage. E.g. --sample-blocks 4 --weeks 3 = 4×3 week "
                             "blocks (winter/spring/summer/autumn).")
    parser.add_argument("--fit-profile", choices=["full", "fast", "fast-refit", "fast_ashp"], default="full",
                        help="Run profile. 'fast' is for quick ASHP-label iteration and skips tank re-fit; "
                             "'fast-refit' keeps a cheap tank fit on sampled seasonal blocks; "
                             "'fast_ashp' runs ASHP-only experiments on a small subset with fixed tank params.")
    args = parser.parse_args()
    main(args.csv, args.yaml, args.output, args.dhw, args.train_frac, args.max_nfev,
         args.weeks, args.start_week, args.sample_blocks, args.fit_profile)