#!/usr/bin/env python3
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

from src import ashp_model, tank_model

DEFAULT_PARAMS_JSON = Path("output/params.json")
DEFAULT_OUTPUT_CSV = Path("output/forecast_tank_predictions.csv")

WATER_CP_KJ_PER_KG_K = 4.186
KJ_TO_KWH = 1.0 / 3600.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run forward tank forecast simulation using trained digital twin parameters."
    )
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_JSON)
    parser.add_argument("--forecast-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)

    parser.add_argument(
        "--init-state",
        type=str,
        default=None,
        help="Optional initial tank state as 'bottom,mid,mid_hi,top' in degC.",
    )
    parser.add_argument(
        "--default-tout-c",
        type=float,
        default=7.0,
        help="Fallback outdoor temperature if not present in forecast CSV.",
    )
    parser.add_argument(
        "--default-ambient-offset-c",
        type=float,
        default=10.0,
        help="Ambient offset above T_out when ambient column is missing.",
    )

    parser.add_argument(
        "--ashp-on-below-c",
        type=float,
        default=48.0,
        help="ASHP turns on when top node <= this threshold.",
    )
    parser.add_argument(
        "--ashp-off-above-c",
        type=float,
        default=55.0,
        help="ASHP turns off when top node >= this threshold.",
    )

    parser.add_argument(
        "--control-mode",
        type=str,
        choices=["normal", "solar_priority", "preheat", "auto"],
        default="normal",
        help=(
            "Control mode for ASHP hysteresis. Use 'auto' for forecast-based mode selection. "
            "Default 'normal' preserves existing behavior."
        ),
    )
    parser.add_argument(
        "--normal-on-below-c",
        type=float,
        default=None,
        help="Optional override for normal mode ON threshold. Defaults to --ashp-on-below-c.",
    )
    parser.add_argument(
        "--normal-off-above-c",
        type=float,
        default=None,
        help="Optional override for normal mode OFF threshold. Defaults to --ashp-off-above-c.",
    )
    parser.add_argument(
        "--solar-priority-on-below-c",
        type=float,
        default=46.0,
        help="Solar-priority ON threshold (lower target to leave solar headroom).",
    )
    parser.add_argument(
        "--solar-priority-off-above-c",
        type=float,
        default=52.0,
        help="Solar-priority OFF threshold (lower top-node charge target).",
    )
    parser.add_argument(
        "--solar-priority-safety-on-below-c",
        type=float,
        default=44.0,
        help="Critical top-node safety threshold for solar_priority: force ASHP ON when top <= this value.",
    )
    parser.add_argument(
        "--preheat-on-below-c",
        type=float,
        default=50.0,
        help="Preheat ON threshold (higher preparedness threshold).",
    )
    parser.add_argument(
        "--preheat-off-above-c",
        type=float,
        default=58.0,
        help="Preheat OFF threshold (higher top-node charge target).",
    )
    parser.add_argument(
        "--high-solar-kwh-threshold",
        type=float,
        default=20.0,
        help="If forecast solar energy sum exceeds this, auto selector chooses solar_priority.",
    )
    parser.add_argument(
        "--high-dhw-kwh-threshold",
        type=float,
        default=12.0,
        help="If forecast DHW energy sum exceeds this, auto selector chooses preheat.",
    )
    parser.add_argument(
        "--cold-ambient-mean-c-threshold",
        type=float,
        default=6.0,
        help="If mean forecast ambient is below this, auto selector chooses preheat.",
    )

    parser.add_argument(
        "--t-hot-c",
        type=float,
        default=50.0,
        help="Delivery temperature for converting DHW energy to litres when needed.",
    )
    parser.add_argument(
        "--t-mains-c",
        type=float,
        default=None,
        help="Optional mains temperature override (degC). Defaults to params tank.T_mains.",
    )
    return parser.parse_args()


def _find_first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _infer_dt_hours(ts: pd.Series) -> float:
    diffs = ts.sort_values().diff().dropna()
    if diffs.empty:
        return 0.5
    return float(diffs.dt.total_seconds().median() / 3600.0)


def _load_params(path: Path) -> tuple[tank_model.TankParams, ashp_model.ASHPParams]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    t = payload["tank"]
    a = payload["ashp"]

    tank_p = tank_model.TankParams(
        UA_loss=np.asarray(t["UA_loss"], dtype=float),
        UA_adj=np.asarray(t["UA_adj"], dtype=float),
        f_st=np.asarray(t["f_st"], dtype=float),
        f_ashp=np.asarray(t["f_ashp"], dtype=float),
        f_imm=np.asarray(t["f_imm"], dtype=float),
        mix_coeff=float(t["mix_coeff"]),
        alpha_draw=float(t["alpha_draw"]),
        T_mains=float(t["T_mains"]),
    )
    ashp_p = ashp_model.ASHPParams(
        a=np.asarray(a["a"], dtype=float),
        b=np.asarray(a["b"], dtype=float),
        c=np.asarray(a["c"], dtype=float),
    )
    return tank_p, ashp_p


def _initial_state(df: pd.DataFrame, t_mains_c: float, init_arg: str | None) -> np.ndarray:
    if init_arg:
        parts = [float(x.strip()) for x in init_arg.split(",")]
        if len(parts) != 4:
            raise ValueError("--init-state must have exactly 4 comma-separated values")
        return np.asarray(parts, dtype=float)

    candidates = [
        ["tank_bottom_sim_c", "tank_mid_sim_c", "tank_mid_hi_sim_c", "tank_top_sim_c"],
        ["tank_bottom_meas_c", "tank_mid_meas_c", "tank_mid_hi_meas_c", "tank_top_meas_c"],
        ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"],
    ]
    for cols in candidates:
        if all(c in df.columns for c in cols):
            return df.loc[df.index[0], cols].to_numpy(dtype=float)

    # Reasonable default profile if no tank-state columns are provided.
    return np.array([t_mains_c + 15.0, t_mains_c + 28.0, t_mains_c + 37.0, t_mains_c + 43.0], dtype=float)


def _dhw_draw_liters(df: pd.DataFrame, t_hot_c: float, t_mains_c: float) -> tuple[np.ndarray, np.ndarray, str]:
    if "dhw_draw_size_l" in df.columns:
        draw_l = pd.to_numeric(df["dhw_draw_size_l"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if "dhw_draw_energy_kwh" in df.columns:
            draw_kwh = pd.to_numeric(df["dhw_draw_energy_kwh"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            delta_t = max(t_hot_c - t_mains_c, 0.0)
            draw_kwh = draw_l * WATER_CP_KJ_PER_KG_K * delta_t * KJ_TO_KWH
        return np.clip(draw_l, 0.0, None), np.clip(draw_kwh, 0.0, None), "dhw_draw_size_l"

    if "dhw_draw_energy_kwh" in df.columns:
        draw_kwh = pd.to_numeric(df["dhw_draw_energy_kwh"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        delta_t = max(t_hot_c - t_mains_c, 1e-9)
        draw_l = draw_kwh / (WATER_CP_KJ_PER_KG_K * delta_t * KJ_TO_KWH)
        return np.clip(draw_l, 0.0, None), np.clip(draw_kwh, 0.0, None), "dhw_draw_energy_kwh"

    return np.zeros(len(df), dtype=float), np.zeros(len(df), dtype=float), "none"


def _resolve_mode_thresholds(args: argparse.Namespace) -> dict[str, tuple[float, float]]:
    normal_on = float(args.normal_on_below_c) if args.normal_on_below_c is not None else float(args.ashp_on_below_c)
    normal_off = float(args.normal_off_above_c) if args.normal_off_above_c is not None else float(args.ashp_off_above_c)
    return {
        "normal": (normal_on, normal_off),
        "solar_priority": (float(args.solar_priority_on_below_c), float(args.solar_priority_off_above_c)),
        "preheat": (float(args.preheat_on_below_c), float(args.preheat_off_above_c)),
    }


def _select_control_mode(
    args: argparse.Namespace,
    q_st_kwh: np.ndarray,
    q_dhw_kwh: np.ndarray,
    t_amb_c: np.ndarray,
) -> tuple[str, str]:
    if args.control_mode != "auto":
        return str(args.control_mode), f"manual:{args.control_mode}"

    solar_sum = float(np.nansum(q_st_kwh))
    dhw_sum = float(np.nansum(q_dhw_kwh))
    amb_mean = float(np.nanmean(t_amb_c)) if len(t_amb_c) > 0 else np.nan

    if solar_sum >= float(args.high_solar_kwh_threshold):
        return "solar_priority", f"auto:high_solar(sum={solar_sum:.2f}kWh)"
    if (dhw_sum >= float(args.high_dhw_kwh_threshold)) or (np.isfinite(amb_mean) and amb_mean <= float(args.cold_ambient_mean_c_threshold)):
        reason = f"auto:high_dhw(sum={dhw_sum:.2f}kWh)" if dhw_sum >= float(args.high_dhw_kwh_threshold) else f"auto:cold_ambient(mean={amb_mean:.2f}C)"
        return "preheat", reason
    return "normal", "auto:default_normal"


def main() -> None:
    args = parse_args()

    tank_p, ashp_p = _load_params(args.params)
    t_mains_c = float(args.t_mains_c) if args.t_mains_c is not None else float(tank_p.T_mains)

    fc = pd.read_csv(args.forecast_csv)
    if "timestamp" not in fc.columns:
        raise ValueError("Forecast CSV must include 'timestamp'")

    fc["timestamp"] = pd.to_datetime(fc["timestamp"], errors="raise")
    fc = fc.sort_values("timestamp").reset_index(drop=True)

    dt_h = _infer_dt_hours(fc["timestamp"])
    dt_s = dt_h * 3600.0

    tout_col = _find_first_column(fc, ["t_out_c", "T_out [C]", "Tout", "tout_c", "outdoor_temp_c"])
    amb_col = _find_first_column(fc, ["t_amb_c", "ambient_c", "T_amb [C]", "ambient_temp_c"])

    if tout_col is None:
        T_out = np.full(len(fc), float(args.default_tout_c), dtype=float)
        tout_source = "default"
    else:
        T_out = pd.to_numeric(fc[tout_col], errors="coerce").fillna(args.default_tout_c).to_numpy(dtype=float)
        tout_source = tout_col

    if amb_col is None:
        T_amb = T_out + float(args.default_ambient_offset_c)
        amb_source = "T_out + offset"
    else:
        T_amb = pd.to_numeric(fc[amb_col], errors="coerce").fillna(np.nan).to_numpy(dtype=float)
        # Fill gaps from T_out+offset.
        fallback_amb = T_out + float(args.default_ambient_offset_c)
        T_amb = np.where(np.isfinite(T_amb), T_amb, fallback_amb)
        amb_source = amb_col

    st_kwh_col = _find_first_column(fc, ["st_kwh", "solar_kwh", "st_energy_kwh"])
    st_kw_col = _find_first_column(fc, ["st_power_kw", "solar_power_kw", "ST Power [kW]"])
    if st_kwh_col is not None:
        Q_st = pd.to_numeric(fc[st_kwh_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        st_source = st_kwh_col
    elif st_kw_col is not None:
        st_kw = pd.to_numeric(fc[st_kw_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        Q_st = st_kw * dt_h
        st_source = f"{st_kw_col}*dt"
    else:
        Q_st = np.zeros(len(fc), dtype=float)
        st_source = "none"

    V_draw, Q_dhw_in_kwh, dhw_source = _dhw_draw_liters(fc, t_hot_c=args.t_hot_c, t_mains_c=t_mains_c)

    mode_thresholds = _resolve_mode_thresholds(args)
    selected_mode, mode_reason = _select_control_mode(args, q_st_kwh=Q_st, q_dhw_kwh=Q_dhw_in_kwh, t_amb_c=T_amb)
    if selected_mode not in mode_thresholds:
        raise ValueError(f"Unsupported selected control mode: {selected_mode}")
    mode_on_below_c, mode_off_above_c = mode_thresholds[selected_mode]
    solar_safety_on_below_c = float(args.solar_priority_safety_on_below_c)

    ashp_on_col = _find_first_column(fc, ["ashp_on", "ashp_enable", "hp_on"])
    ashp_on_input = None
    if ashp_on_col is not None:
        ashp_on_input = pd.to_numeric(fc[ashp_on_col], errors="coerce").fillna(0).to_numpy(dtype=int) > 0

    T = _initial_state(fc, t_mains_c=t_mains_c, init_arg=args.init_state)

    pred_rows: list[dict] = []
    ashp_on = False
    safety_override_count = 0

    for i in range(len(fc)):
        ts = fc.loc[i, "timestamp"]
        top_now = float(T[3])

        if ashp_on_input is not None:
            ashp_on = bool(ashp_on_input[i])
        else:
            # Safety override for solar-priority control: avoid deep temperature drops.
            if selected_mode == "solar_priority" and top_now <= solar_safety_on_below_c:
                ashp_on = True
                safety_override_count += 1
            else:
                if ashp_on and top_now >= mode_off_above_c:
                    ashp_on = False
                elif (not ashp_on) and top_now <= mode_on_below_c:
                    ashp_on = True

        if ashp_on:
            T_sink = float(ashp_model.sink_proxy(T[0], T[1]))
            q_ashp_kw = float(ashp_model.predict_capacity(T_out[i], T_sink, ashp_p))
            q_ashp_kwh = max(0.0, q_ashp_kw * dt_h)
        else:
            q_ashp_kwh = 0.0

        T_next = tank_model.tank_step(
            T,
            Q_st_kwh=float(Q_st[i]),
            Q_ashp_kwh=float(q_ashp_kwh),
            Q_imm_kwh=0.0,
            T_amb=float(T_amb[i]),
            V_draw_l=float(V_draw[i]),
            T_cold=float(t_mains_c),
            params=tank_p,
            dt_s=float(dt_s),
            Q_dhw_kwh=0.0,
        )

        pred_rows.append(
            {
                "timestamp": ts,
                "tank_bottom_pred_c": float(T_next[0]),
                "tank_mid_pred_c": float(T_next[1]),
                "tank_mid_hi_pred_c": float(T_next[2]),
                "tank_top_pred_c": float(T_next[3]),
                "predicted_ashp_heat_kwh": float(q_ashp_kwh),
                "predicted_dhw_draw_l": float(V_draw[i]),
                "predicted_dhw_draw_energy_kwh": float(Q_dhw_in_kwh[i]),
                "selected_control_mode": selected_mode,
            }
        )

        T = T_next

    out_df = pd.DataFrame(pred_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    print("Forecast simulation complete")
    print(f"Forecast CSV: {args.forecast_csv}")
    print(f"Output CSV: {args.output}")
    print(f"Rows simulated: {len(out_df)}")
    print(f"dt_h inferred: {dt_h:.3f}")
    print(f"T_out source: {tout_source}")
    print(f"Ambient source: {amb_source}")
    print(f"Solar source: {st_source}")
    print(f"DHW source: {dhw_source}")
    print(f"Selected control mode: {selected_mode} ({mode_reason})")
    print(f"Mode thresholds [on<=, off>=] [degC]: {mode_on_below_c:.1f}, {mode_off_above_c:.1f}")
    if selected_mode == "solar_priority":
        print(
            f"Solar-priority safety override [force on at top<=] [degC]: {solar_safety_on_below_c:.1f}"
        )
        print(f"Solar-priority safety override activations: {safety_override_count}")
    print(f"Total predicted ASHP heat [kWh]: {out_df['predicted_ashp_heat_kwh'].sum():.2f}")
    print(f"Total predicted DHW draw [L]: {out_df['predicted_dhw_draw_l'].sum():.2f}")


if __name__ == "__main__":
    main()
