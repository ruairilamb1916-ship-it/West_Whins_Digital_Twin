#!/usr/bin/env python3
"""Inspect fitted ASHP maps against measured operating points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import ashp_model, data_loader


def _load_params(params_path: Path) -> ashp_model.ASHPParams:
    with open(params_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    a = np.asarray(payload["ashp"]["a"], dtype=float)
    b = np.asarray(payload["ashp"]["b"], dtype=float)
    c = np.asarray(payload["ashp"].get("c", ashp_model.ASHPParams().c), dtype=float)
    return ashp_model.ASHPParams(a=a, b=b, c=c)


def _select_fast_train_subset(df: pd.DataFrame, labels_path: Path) -> pd.DataFrame:
    if not labels_path.exists():
        return df

    labels = pd.read_csv(labels_path)
    if "timestamp" not in labels.columns:
        return df

    ts = pd.to_datetime(labels["timestamp"], errors="coerce")
    ts = pd.DatetimeIndex(ts.dropna().unique())
    if len(ts) == 0:
        return df

    common = df.index.intersection(ts)
    if len(common) == 0:
        return df
    return df.loc[common].copy()


def _load_q_mix_aligned(index: pd.DatetimeIndex, labels_path: Path) -> pd.Series | None:
    if not labels_path.exists():
        return None

    labels = pd.read_csv(labels_path)
    if "timestamp" not in labels.columns or "q_mix" not in labels.columns:
        return None

    labels["timestamp"] = pd.to_datetime(labels["timestamp"], errors="coerce")
    labels = labels.dropna(subset=["timestamp"])
    if labels.empty:
        return None

    q_mix = pd.Series(labels["q_mix"].to_numpy(dtype=float), index=pd.DatetimeIndex(labels["timestamp"]))
    q_mix = q_mix.groupby(level=0).last()
    return q_mix.reindex(index)


def _load_q_backcalc_aligned(index: pd.DatetimeIndex, labels_path: Path) -> pd.Series | None:
    if not labels_path.exists():
        return None

    labels = pd.read_csv(labels_path)
    if "timestamp" not in labels.columns or "Q_meas_backcalc_kwh" not in labels.columns:
        return None

    labels["timestamp"] = pd.to_datetime(labels["timestamp"], errors="coerce")
    labels = labels.dropna(subset=["timestamp"])
    if labels.empty:
        return None

    q_back = pd.Series(labels["Q_meas_backcalc_kwh"].to_numpy(dtype=float), index=pd.DatetimeIndex(labels["timestamp"]))
    q_back = q_back.groupby(level=0).last()
    return q_back.reindex(index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug fitted ASHP map outputs.")
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "FullDS_Findhorn.csv")
    parser.add_argument("--yaml", type=Path, default=ROOT / "column_mapping.yaml")
    parser.add_argument("--params", type=Path, default=ROOT / "output" / "params.json")
    parser.add_argument("--labels", type=Path, default=ROOT / "output" / "debug_labels_fast.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "debug_ashp_map.csv")
    parser.add_argument("--start-week", type=int, default=0, help="Start week offset (0-indexed)")
    parser.add_argument("--weeks", type=int, default=12, help="Number of weeks to process")
    args = parser.parse_args()

    p = _load_params(args.params)

    df = data_loader.load_and_clean(args.csv, args.yaml)

    start_row = args.start_week * 7 * 48
    n_rows = args.weeks * 7 * 48
    df = df.iloc[start_row:start_row + n_rows].copy()

    df = _select_fast_train_subset(df, args.labels)

    if df.empty:
        raise ValueError(
            f"Empty dataframe after subsetting to weeks {args.start_week}–{args.start_week + args.weeks} "
            f"and aligning to {args.labels}. Check subset bounds and debug_labels_fast.csv coverage."
        )

    required = ["t_out_c", "tank_bottom_c", "tank_mid_c", "tank_top_c", "ashp_inst_kwh"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    t_out = df["t_out_c"].to_numpy(dtype=float)
    t_sink = ashp_model.sink_proxy(
        df["tank_bottom_c"].to_numpy(dtype=float),
        df["tank_mid_c"].to_numpy(dtype=float),
    )

    p_meas = df["ashp_inst_kwh"].fillna(0).to_numpy(dtype=float)
    p_fit = ashp_model.predict_power(t_out, t_sink, p)
    cop_fit = ashp_model.predict_cop(t_out, t_sink, p)
    q_fit = p_fit * cop_fit

    cop_target = None
    q_mix_aligned = _load_q_mix_aligned(df.index, args.labels)
    if q_mix_aligned is not None:
        q_mix = q_mix_aligned.to_numpy(dtype=float)
        cop_target_arr = np.full(len(df), np.nan, dtype=float)
        valid_target = np.isfinite(q_mix) & (p_meas > 0.05)
        cop_target_arr[valid_target] = q_mix[valid_target] / p_meas[valid_target]
        cop_target = cop_target_arr

    q_backcalc = None
    cop_backcalc = None
    q_backcalc_aligned = _load_q_backcalc_aligned(df.index, args.labels)
    if q_backcalc_aligned is not None:
        q_backcalc = q_backcalc_aligned.to_numpy(dtype=float)
        cop_backcalc_arr = np.full(len(df), np.nan, dtype=float)
        valid_backcalc = np.isfinite(q_backcalc) & (p_meas > 0.05)
        cop_backcalc_arr[valid_backcalc] = q_backcalc[valid_backcalc] / p_meas[valid_backcalc]
        cop_backcalc = cop_backcalc_arr

    out_df = pd.DataFrame(
        {
            "timestamp": df.index,
            "t_out_c": df["t_out_c"].to_numpy(dtype=float),
            "tank_bottom_c": df["tank_bottom_c"].to_numpy(dtype=float),
            "tank_mid_c": df["tank_mid_c"].to_numpy(dtype=float),
            "tank_top_c": df["tank_top_c"].to_numpy(dtype=float),
            "P_meas": p_meas,
            "P_fit": p_fit,
            "Q_fit": q_fit,
            "COP_fit": cop_fit,
        }
    )
    if cop_target is not None:
        out_df["COP_target"] = cop_target
    if q_backcalc is not None:
        out_df["Q_meas_backcalc_kwh"] = q_backcalc

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print(f"Saved map debug CSV: {args.out}")
    print(f"mean P_fit: {np.nanmean(p_fit):.6f}")
    print(f"mean Q_fit: {np.nanmean(q_fit):.6f}")
    print(f"mean COP_fit: {np.nanmean(cop_fit):.6f}")
    print(f"median COP_fit: {np.nanmedian(cop_fit):.6f}")
    print(f"frac COP_fit > 3: {np.nanmean(cop_fit > 3.0):.6f}")
    if cop_target is not None:
        valid_target = np.isfinite(cop_target)
        if np.any(valid_target):
            ct = cop_target[valid_target]
            print(f"mean COP_target: {np.nanmean(ct):.6f}")
            print(f"median COP_target: {np.nanmedian(ct):.6f}")
            print(f"frac COP_target > 2: {np.nanmean(ct > 2.0):.6f}")
            print(f"frac COP_target > 3: {np.nanmean(ct > 3.0):.6f}")
            print(f"frac COP_target > 4: {np.nanmean(ct > 4.0):.6f}")
    if cop_backcalc is not None:
        valid_backcalc = np.isfinite(cop_backcalc)
        if np.any(valid_backcalc):
            cb = cop_backcalc[valid_backcalc]
            print(f"mean COP_backcalc: {np.nanmean(cb):.6f}")
            print(f"median COP_backcalc: {np.nanmedian(cb):.6f}")


if __name__ == "__main__":
    main()