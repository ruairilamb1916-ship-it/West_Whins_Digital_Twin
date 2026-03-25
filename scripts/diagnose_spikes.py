#!/usr/bin/env python3
"""Diagnose closed-loop temperature spikes from fitted params.json."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import ashp_model, data_loader, identification, solar_thermal, tank_model


LABELS = ["bottom", "mid", "mid_hi", "top"]


def load_params(params_path: Path) -> tuple[tank_model.TankParams, ashp_model.ASHPParams]:
    saved = json.loads(params_path.read_text())

    tank_p = tank_model.TankParams()
    tank_p.UA_loss = np.array(saved["tank"]["UA_loss"], dtype=float)
    tank_p.UA_adj = np.array(saved["tank"]["UA_adj"], dtype=float)
    tank_p.f_st = np.array(saved["tank"]["f_st"], dtype=float)
    tank_p.f_ashp = np.array(saved["tank"]["f_ashp"], dtype=float)
    tank_p.f_imm = np.array(saved["tank"]["f_imm"], dtype=float)
    tank_p.mix_coeff = float(saved["tank"]["mix_coeff"])
    tank_p.alpha_draw = float(saved["tank"]["alpha_draw"])
    tank_p.T_mains = float(saved["tank"]["T_mains"])

    ashp_p = ashp_model.ASHPParams(
        a=np.array(saved["ashp"]["a"], dtype=float),
        b=np.array(saved["ashp"]["b"], dtype=float),
    )
    return tank_p, ashp_p


def main() -> None:
    root = ROOT
    tank_p, ashp_p = load_params(root / "output" / "params.json")

    df = data_loader.load_and_clean(root / "data" / "FullDS_Findhorn.csv", root / "column_mapping.yaml")
    df = df.dropna(subset=["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"], how="all").copy()
    df["st_kwh"] = solar_thermal.compute_st_energy(df)

    split_idx = int(len(df) * 0.7)
    df_val = df.iloc[split_idx:].copy()
    inputs = identification.prepare_inputs(df_val, ashp_p)

    T_cl = identification.simulate_closed_loop(
        inputs["T_meas"][0],
        inputs["Q_st"],
        inputs["Q_imm"],
        inputs["T_amb"],
        inputs["V_draw"],
        inputs["T_cold"],
        inputs["T_out"],
        inputs["P_meas"],
        ashp_p,
        tank_p,
    )

    T_nodes = T_cl[1 : len(df_val) + 1]
    T_meas = inputs["T_meas"]

    rmse_cl = float(np.sqrt(np.nanmean((T_nodes - T_meas) ** 2)))
    print(f"rmse_closed_loop_c={rmse_cl:.3f}")
    print(f"t_cl_min_c={float(np.nanmin(T_nodes)):.3f}")
    print(f"t_cl_max_c={float(np.nanmax(T_nodes)):.3f}")

    idx = int(np.nanargmax(T_nodes))
    row, col = np.unravel_index(idx, T_nodes.shape)

    print(f"max_temp_c={T_nodes[row, col]:.3f}")
    print(f"max_node={LABELS[col]}")
    print(f"max_timestamp={df_val.index[row]}")

    print("top10_spikes:")
    flat_idx = np.argpartition(T_nodes.ravel(), -10)[-10:]
    flat_idx = flat_idx[np.argsort(T_nodes.ravel()[flat_idx])[::-1]]
    for fi in flat_idx:
        rr, cc = np.unravel_index(int(fi), T_nodes.shape)
        print(f"{df_val.index[rr]} node={LABELS[cc]} temp={T_nodes[rr, cc]:.2f}")

    print("context_around_peak:")
    start = max(0, row - 4)
    end = min(len(df_val) - 1, row + 4)
    for rr in range(start, end + 1):
        print(
            f"{df_val.index[rr]} | "
            f"Tbot={T_nodes[rr, 0]:.2f} Tmid={T_nodes[rr, 1]:.2f} "
            f"Tmhi={T_nodes[rr, 2]:.2f} Ttop={T_nodes[rr, 3]:.2f} | "
            f"P={inputs['P_meas'][rr]:.3f} T_out={inputs['T_out'][rr]:.1f} "
            f"Q_st={inputs['Q_st'][rr]:.3f} Q_imm={inputs['Q_imm'][rr]:.3f} "
            f"V_draw={inputs['V_draw'][rr]:.1f}"
        )


if __name__ == "__main__":
    main()
