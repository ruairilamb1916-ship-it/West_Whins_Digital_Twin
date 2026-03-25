#!/usr/bin/env python3
"""
scripts/check_label_overlap.py
Print and save scalar comparison metrics between the heuristic and
energy-balance ASHP heat labels.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data_loader, identification  # noqa: E402

logging.disable(logging.CRITICAL)  # suppress module-level logs; only scalars are printed

CSV_PATH = ROOT / "data" / "FullDS_Findhorn.csv"
YAML_PATH = ROOT / "column_mapping.yaml"
OUT_PATH = ROOT / "output" / "label_overlap.json"


def main() -> None:
    df = data_loader.load_and_clean(CSV_PATH, YAML_PATH)

    q_heur = identification.back_calculate_ashp_heat(df)
    q_eb = identification.back_calculate_ashp_heat_energy_balance(df)

    h = np.isfinite(q_heur.to_numpy(dtype=float))
    e = np.isfinite(q_eb.to_numpy(dtype=float))

    heuristic_only_count = int((h & ~e).sum())
    eb_only_count = int((~h & e).sum())
    both_count = int((h & e).sum())
    neither_count = int((~h & ~e).sum())

    overlap = h & e
    q_h_ov = q_heur.to_numpy(dtype=float)[overlap]
    q_e_ov = q_eb.to_numpy(dtype=float)[overlap]

    if overlap.sum() > 0:
        median_q_old_overlap = float(np.median(q_h_ov))
        median_q_eb_overlap = float(np.median(q_e_ov))
        ratios = q_e_ov / np.where(q_h_ov > 0, q_h_ov, np.nan)
        median_ratio_eb_to_old = float(np.nanmedian(ratios))
    else:
        median_q_old_overlap = float("nan")
        median_q_eb_overlap = float("nan")
        median_ratio_eb_to_old = float("nan")

    # Charging mask: same criteria used by the heuristic labeller
    ashp_on = df["ashp_inst_kwh"].fillna(0) > 0.05
    hx_on = df["tank_top_c"].fillna(0).diff() > 0.05
    imm_off = df["imm_tot_inst_kwh"].fillna(0) < 0.01
    st_col = "st_kwh"
    st_low = (
        df[st_col].fillna(0) < 0.05
        if st_col in df.columns
        else pd.Series(True, index=df.index)
    )
    charging = (ashp_on & hx_on & imm_off & st_low).to_numpy()
    charging_overlap_count = int((overlap & charging).sum())

    results = {
        "heuristic_only_count": heuristic_only_count,
        "eb_only_count": eb_only_count,
        "both_count": both_count,
        "neither_count": neither_count,
        "median_q_old_overlap": median_q_old_overlap,
        "median_q_eb_overlap": median_q_eb_overlap,
        "median_ratio_eb_to_old": median_ratio_eb_to_old,
        "charging_overlap_count": charging_overlap_count,
    }

    for key, val in results.items():
        print(f"{key}: {val}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved to {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
