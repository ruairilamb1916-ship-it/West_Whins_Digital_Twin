#!/usr/bin/env python3
"""Diagnostic plot: measured vs model over a short time window.

Usage examples
--------------
# Default: first 7 days of validation set
python scripts/plot_window.py

# Custom start date and length
python scripts/plot_window.py --start 2023-11-01 --days 5

Produces output/plots/window_<start>_<days>d.png with:
  - Row 1-4 : tank temperatures (bottom → top): measured, teacher-forced, closed-loop
  - Row 5   : ASHP electrical power + inferred heat delivery
  - Row 6   : solar-thermal and immersion heat
  - Row 7   : inferred draw-off volume
  - Row 8   : outdoor and ambient temperatures
  - Row 9   : per-step closed-loop temperature error (top node)

This gives a compact view for spotting *when* the model diverges:
  - ASHP on? (rows 1-4 vs row 5)
  - Standing-loss creep? (rows 1-4 during idle periods)
  - Draw events misidentified? (row 7 vs row 1-4)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import ashp_model, data_loader, identification, solar_thermal, tank_model


LABELS = ["bottom", "mid", "mid-hi", "top"]
NODE_COLOURS = ["#4575b4", "#74add1", "#f46d43", "#d73027"]


def load_params(params_path: Path) -> tuple[tank_model.TankParams, ashp_model.ASHPParams]:
    saved = json.loads(params_path.read_text())
    tank_p = tank_model.TankParams()
    tank_p.UA_loss    = np.array(saved["tank"]["UA_loss"], dtype=float)
    tank_p.UA_adj     = np.array(saved["tank"]["UA_adj"], dtype=float)
    tank_p.f_st       = np.array(saved["tank"]["f_st"], dtype=float)
    tank_p.f_ashp     = np.array(saved["tank"]["f_ashp"], dtype=float)
    tank_p.f_imm      = np.array(saved["tank"]["f_imm"], dtype=float)
    tank_p.mix_coeff  = float(saved["tank"]["mix_coeff"])
    tank_p.alpha_draw = float(saved["tank"]["alpha_draw"])
    tank_p.T_mains    = float(saved["tank"]["T_mains"])

    ashp_p = ashp_model.ASHPParams(
        a=np.array(saved["ashp"]["a"], dtype=float),
        b=np.array(saved["ashp"]["b"], dtype=float),
    )
    return tank_p, ashp_p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Short-window diagnostic plot")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date (YYYY-MM-DD). Defaults to first day of validation set.")
    parser.add_argument("--days",  type=int, default=7,
                        help="Number of days to plot (default 7).")
    parser.add_argument("--train", action="store_true",
                        help="Use training set instead of validation set.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tank_p, ashp_p = load_params(ROOT / "output" / "params.json")

    df = data_loader.load_and_clean(
        ROOT / "data" / "FullDS_Findhorn.csv",
        ROOT / "column_mapping.yaml",
    )
    df = df.dropna(
        subset=["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"],
        how="all",
    ).copy()
    df["st_kwh"] = solar_thermal.compute_st_energy(df)

    split_idx = int(len(df) * 0.7)
    df_work = df.iloc[:split_idx].copy() if args.train else df.iloc[split_idx:].copy()
    set_label = "train" if args.train else "val"

    # ── Select window ─────────────────────────────────────────────────────────
    if args.start is not None:
        t_start = pd.Timestamp(args.start, tz=df_work.index.tz)
    else:
        t_start = df_work.index[0]

    t_end = t_start + pd.Timedelta(days=args.days)
    mask_w = (df_work.index >= t_start) & (df_work.index < t_end)
    df_w = df_work.loc[mask_w].copy()

    if len(df_w) < 4:
        print(f"ERROR: only {len(df_w)} rows in requested window. "
              f"Available range: {df_work.index[0]} – {df_work.index[-1]}")
        sys.exit(1)

    print(f"Window: {df_w.index[0]}  →  {df_w.index[-1]}  ({len(df_w)} steps)")

    # ── Prepare inputs for the FULL working set so closed-loop state is correct ─
    # We run CL from the start of the full set so state is warm, then slice.
    inputs_full = identification.prepare_inputs(df_work, ashp_p)

    # Teacher-forced over full set
    T_meas_full = inputs_full["T_meas"]
    N_full = len(T_meas_full)
    T_tf_full = np.zeros_like(T_meas_full)
    T_tf_full[0] = T_meas_full[0]
    for k in range(N_full - 1):
        T_tf_full[k + 1] = tank_model.tank_step(
            T_meas_full[k],
            float(inputs_full["Q_st"][k]),
            float(inputs_full["Q_ashp"][k]),
            float(inputs_full["Q_imm"][k]),
            float(inputs_full["T_amb"][k]),
            float(inputs_full["V_draw"][k]),
            float(inputs_full["T_cold"][k]),
            tank_p,
        )

    # Closed-loop over full set
    T_cl_full = identification.simulate_closed_loop(
        T_meas_full[0],
        inputs_full["Q_st"],
        inputs_full["Q_imm"],
        inputs_full["T_amb"],
        inputs_full["V_draw"],
        inputs_full["T_cold"],
        inputs_full["T_out"],
        inputs_full["P_meas"],
        ashp_p,
        tank_p,
    )
    # T_cl_full is (N+1, 4); index 0 is initial state, index k+1 corresponds to step k
    T_cl_states = T_cl_full[1 : N_full + 1]  # (N, 4) aligned with T_meas_full

    # ── Slice to the requested window ─────────────────────────────────────────
    row_start = int(np.searchsorted(df_work.index, t_start))
    row_end   = int(np.searchsorted(df_work.index, t_end))
    row_end   = min(row_end, N_full)

    idx_w   = slice(row_start, row_end)
    t_index = df_work.index[idx_w]
    T_meas  = T_meas_full[idx_w]
    T_tf    = T_tf_full[idx_w]
    T_cl    = T_cl_states[idx_w]

    Q_st    = inputs_full["Q_st"][idx_w]
    Q_ashp  = inputs_full["Q_ashp"][idx_w]
    Q_imm   = inputs_full["Q_imm"][idx_w]
    P_meas  = inputs_full["P_meas"][idx_w]
    V_draw  = inputs_full["V_draw"][idx_w]
    T_out   = inputs_full["T_out"][idx_w]
    T_amb   = inputs_full["T_amb"][idx_w]

    # ── Build figure ──────────────────────────────────────────────────────────
    n_rows = 9
    fig, axs = plt.subplots(n_rows, 1, figsize=(16, 3.0 * n_rows), sharex=True)
    fig.subplots_adjust(hspace=0.35)

    # Shade ASHP-on periods on every axis
    ashp_on = P_meas > 0.02
    ashp_starts = t_index[np.where(np.diff(ashp_on.astype(int), prepend=0) == 1)[0]]
    ashp_ends   = t_index[np.where(np.diff(ashp_on.astype(int), append=0) == -1)[0]]

    def shade_ashp(ax):
        for ts, te in zip(ashp_starts, ashp_ends):
            ax.axvspan(ts, te, color="lightgreen", alpha=0.25, lw=0)

    # Rows 0–3: tank node temperatures
    for i in range(4):
        ax = axs[i]
        shade_ashp(ax)
        ax.plot(t_index, T_meas[:, i], color="black",       lw=1.0,  label="measured",        alpha=0.9, zorder=4)
        ax.plot(t_index, T_tf[:,   i], color="steelblue",   lw=1.0,  label="teacher-forced",  linestyle="--", alpha=0.85, zorder=3)
        ax.plot(t_index, T_cl[:,   i], color="tomato",      lw=1.0,  label="closed-loop",     linestyle=":",  alpha=0.85, zorder=3)
        rmse_tf = float(np.sqrt(np.nanmean((T_tf[:, i] - T_meas[:, i]) ** 2)))
        rmse_cl = float(np.sqrt(np.nanmean((T_cl[:, i] - T_meas[:, i]) ** 2)))
        ax.set_title(
            f"Node: {LABELS[i]}  |  TF RMSE {rmse_tf:.2f} °C  |  CL RMSE {rmse_cl:.2f} °C",
            fontsize=9, loc="left",
        )
        ax.set_ylabel("°C", fontsize=8)
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.grid(True, alpha=0.2)

    # Row 4: ASHP electrical power + map-inferred heat
    ax = axs[4]
    shade_ashp(ax)
    ax.fill_between(t_index, P_meas * 2, alpha=0.4, color="green",  label="P_elec × 2  (kWh, ×2 for visibility)")
    ax.fill_between(t_index, Q_ashp,     alpha=0.5, color="orange", label="Q_ashp (map-inferred, kWh)")
    ax.set_title("ASHP: electrical input (×2) and map-inferred heat", fontsize=9, loc="left")
    ax.set_ylabel("kWh / interval", fontsize=8)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.2)

    # Row 5: ST and immersion heat
    ax = axs[5]
    ax.fill_between(t_index, Q_st,  alpha=0.6, color="gold",    label="Q_st  (kWh)")
    ax.fill_between(t_index, Q_imm, alpha=0.6, color="crimson", label="Q_imm (kWh)")
    ax.set_title("Solar-thermal and immersion heat", fontsize=9, loc="left")
    ax.set_ylabel("kWh / interval", fontsize=8)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.2)

    # Row 6: draw-off volume
    ax = axs[6]
    ax.bar(t_index, V_draw, width=pd.Timedelta(minutes=25), color="purple", alpha=0.6, label="V_draw (L)")
    ax.set_title("Inferred draw-off volume", fontsize=9, loc="left")
    ax.set_ylabel("L / interval", fontsize=8)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.2)

    # Row 7: outdoor and ambient temperature
    ax = axs[7]
    ax.plot(t_index, T_out, color="navy",   lw=0.9, label="T_out (outdoor air)")
    ax.plot(t_index, T_amb, color="teal",   lw=0.9, label="T_amb (plant room)", linestyle="--")
    ax.set_title("Outdoor and plant-room ambient temperature", fontsize=9, loc="left")
    ax.set_ylabel("°C", fontsize=8)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.2)

    # Row 8: closed-loop error (top node) — most visible node for drift
    cl_err_top = T_cl[:, 3] - T_meas[:, 3]
    ax = axs[8]
    shade_ashp(ax)
    ax.axhline(0, color="black", lw=0.7, linestyle="--")
    ax.fill_between(t_index, cl_err_top, 0,
                    where=(cl_err_top >= 0), color="tomato",    alpha=0.5, label="CL over-predicts top")
    ax.fill_between(t_index, cl_err_top, 0,
                    where=(cl_err_top <  0), color="steelblue", alpha=0.5, label="CL under-predicts top")
    ax.set_title("Closed-loop error: top node  (model − measured) [°C]", fontsize=9, loc="left")
    ax.set_ylabel("°C", fontsize=8)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.2)

    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    axs[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=0, ha="center")

    title = (
        f"Diagnostic window  |  {set_label} set  |  "
        f"{t_index[0].strftime('%Y-%m-%d')} — {t_index[-1].strftime('%Y-%m-%d')}  "
        f"({args.days} day{'s' if args.days != 1 else ''})\n"
        "Green shading = ASHP running"
    )
    fig.suptitle(title, fontsize=11, y=1.001)

    out_name = f"window_{t_index[0].strftime('%Y%m%d')}_{args.days}d_{set_label}.png"
    out_path = ROOT / "output" / "plots" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
