"""
Parameter identification for the Stage-1 digital twin.

Two-step procedure:
  1. Fit ASHP maps on intervals with no immersion and low ST.
  2. Fit tank parameters (with ASHP heat derived from the map).

Joint refinement with regularisation is also available.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import ashp_model, solar_thermal, tank_model
from .tank_model import CP, NODE_CAP, NODE_VOL_L, RHO, TankParams

logger = logging.getLogger(__name__)


@dataclass
class IdentificationResult:
    """Container for fitted parameters and diagnostics."""
    tank_params: tank_model.TankParams
    ashp_params: ashp_model.ASHPParams
    hx_effectiveness: float
    cost_history: list[float]


def compute_ashp_heat_kwh(
    P_meas_kwh: np.ndarray | float,
    T_out: np.ndarray | float,
    T_sink: np.ndarray | float,
    ashp_p: ashp_model.ASHPParams,
    dt_h: float = 0.5,
) -> np.ndarray:
    """Compute ASHP delivered heat [kWh], capped by fitted capacity map.

    Heat inferred from ``P * COP`` can exceed the identified capacity map,
    especially during cold/high-sink conditions, which causes unrealistically
    large injections and hot-node spikes in long rollouts.  This helper keeps
    ASHP heat physically bounded while preserving COP dependence.
    """
    P = np.asarray(P_meas_kwh, dtype=float)
    T_a = np.asarray(T_out, dtype=float)
    T_s = np.asarray(T_sink, dtype=float)

    cop = ashp_model.predict_cop(T_a, T_s, ashp_p)
    q_from_cop = P * cop
    q_cap = ashp_model.predict_capacity(T_a, T_s, ashp_p) * dt_h
    return np.clip(np.minimum(q_from_cop, q_cap), 0.0, None)


def back_calculate_ashp_heat(
    df: pd.DataFrame,
    st_col: str = "st_kwh",
    dt_s: float = 1800.0,
) -> pd.Series:
    """Back-calculate ASHP heat delivery [kWh] for ASHP-only intervals.

    Returns a Series of length len(df) with NaN for non-ASHP-only intervals.

    An interval is considered ASHP-only if:
      - ashp_inst_kwh > 0.05  (ASHP was running)
      - imm_tot_inst_kwh < 0.01  (immersion heater was off)
      - st_kwh < 0.05  (negligible solar thermal)
      - all four tank temperature columns are finite at both this
        row and the previous row

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with tank temperatures, ASHP, immersion, and ST data.
    st_col : str
        Name of the solar-thermal energy column.
    dt_s : float
        Interval length in seconds (default 1800).

    Returns
    -------
    pd.Series
        Back-calculated ASHP heat [kWh]; NaN for non-ASHP-only intervals.
    """
    tank_cols = ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]

    ashp_on = df["ashp_inst_kwh"].fillna(0) > 0.05
    hx_on = df["tank_top_c"].fillna(0).diff() > 0.05  # rising top temp indicates heat being delivered to DHW tank not SH
    imm_off = df["imm_tot_inst_kwh"].fillna(0) < 0.01
    st_low = df[st_col].fillna(0) < 0.05 if st_col in df.columns else pd.Series(True, index=df.index)

    # All four tank temps must be finite at this row and the previous row
    T = df[tank_cols].values
    finite_now = np.all(np.isfinite(T), axis=1)
    finite_prev = np.roll(finite_now, 1)
    finite_prev[0] = False  # first row has no predecessor

    mask = ashp_on & hx_on & imm_off & st_low & pd.Series(finite_now & finite_prev, index=df.index)

    n_ashp_only = mask.sum()
    logger.info("ASHP-only intervals found: %d", n_ashp_only)

    Q_back = pd.Series(np.nan, index=df.index)

    if n_ashp_only < 50:
        logger.warning(
            "Insufficient ASHP-only intervals (%d < 50) for back-calculation; "
            "fallback to a = b * 3.0 retained.",
            n_ashp_only,
        )
        return Q_back

    # Default UA_loss for standing-loss correction
    ua_loss_default = TankParams().UA_loss

    T_amb = df["t_amb_c"].fillna(df["t_amb_c"].median()).values

    idx = np.where(mask.values)[0]
    for k in idx:
        dT_sum = 0.0
        loss_sum = 0.0
        T_avg = 0.0
        for i in range(4):
            dT_sum += T[k, i] - T[k - 1, i]
            T_avg += T[k - 1, i]
            loss_sum += ua_loss_default[i] * (T[k - 1, i] - T_amb[k]) * dt_s
        T_avg /= 4.0

        Q_kJ = NODE_CAP * dT_sum + loss_sum  # kJ/K × K + kJ = kJ
        Q_back.iloc[k] = max(Q_kJ / 3600.0, 0.0)

    return Q_back


def back_calculate_ashp_heat_energy_balance(
    df: pd.DataFrame,
    *,
    st_col: str = "st_kwh",
    dt_s: float = 1800.0,
    ashp_on_threshold_kwh: float = 0.05,
    imm_col: str = "imm_tot_inst_kwh",
    clip_max_kwh: float = 12.0,
    cop_floor: float = 1.5,
    cop_ceiling: float = 4.0,
    min_valid: int = 50,
) -> pd.Series:
    """Back-calculate ASHP heat labels from tank energy balance.

    Labels are generated on ASHP-on intervals only using:

        Q_ashp ~= dE_tank + Q_loss + Q_draw - Q_st - Q_imm

    where dE_tank is 4-node stored-energy change, Q_loss is standing loss to
    ambient, and Q_draw is a lightweight draw correction estimated from top-node
    cooling against mains temperature.

    Because the balance terms are noisy, the raw label can collapse well below
    physically plausible ASHP output during charging. We therefore bound the
    final label by a COP floor/ceiling relative to measured electrical input.
    """
    tank_cols = ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]
    T = df[tank_cols].to_numpy(dtype=float)
    N = len(df)

    finite_now = np.all(np.isfinite(T), axis=1)
    finite_prev = np.roll(finite_now, 1)
    finite_prev[0] = False

    ashp_on = df["ashp_inst_kwh"].fillna(0).to_numpy(dtype=float) > ashp_on_threshold_kwh
    st_kwh = (
        df[st_col].fillna(0).to_numpy(dtype=float)
        if st_col in df.columns
        else np.zeros(N, dtype=float)
    )
    q_imm = (
        df[imm_col].fillna(0).to_numpy(dtype=float)
        if imm_col in df.columns
        else np.zeros(N, dtype=float)
    )
    t_amb = df["t_amb_c"].fillna(df["t_amb_c"].median()).to_numpy(dtype=float)
    t_cold = mains_temp_seasonal(df.index)

    ua = TankParams().UA_loss
    q_labels = np.full(N, np.nan, dtype=float)

    for k in range(1, N):
        if not (ashp_on[k] and finite_now[k] and finite_prev[k]):
            continue

        t_prev = T[k - 1]
        t_now = T[k]

        # Tank stored-energy change [kWh].
        dE_kwh = float(NODE_CAP * np.sum(t_now - t_prev) / 3600.0)

        # Standing losses to ambient [kWh].
        q_loss_kwh = float(np.sum(ua * (t_prev - t_amb[k]) * dt_s) / 3600.0)

        # Base balance (without draw correction).
        q_base = dE_kwh + q_loss_kwh - st_kwh[k] - q_imm[k]

        # Lightweight draw correction from top-node cooling fraction.
        denom = max(float(t_prev[3] - t_cold[k]), 1e-3)
        f_draw = float(np.clip((t_prev[3] - t_now[3]) / denom, 0.0, 1.0))
        v_draw_l = f_draw * NODE_VOL_L

        t_draw = 0.7 * float(t_prev[3]) + 0.3 * float(t_prev[2])
        m_draw_kg = v_draw_l * RHO / 1000.0
        q_draw_kwh = float((m_draw_kg * CP * max(t_draw - t_cold[k], 0.0)) / 3600.0)

        q = q_base + q_draw_kwh
        q_labels[k] = float(np.clip(q, 0.0, clip_max_kwh))

    if "ashp_inst_kwh" in df.columns:
        p_meas = df["ashp_inst_kwh"].fillna(0).to_numpy(dtype=float)
        valid = np.isfinite(q_labels) & (p_meas > ashp_on_threshold_kwh)
        if valid.any():
            q_min = cop_floor * p_meas
            q_max = cop_ceiling * p_meas
            q_labels[valid] = np.clip(q_labels[valid], q_min[valid], q_max[valid])

    out = pd.Series(q_labels, index=df.index, name="Q_ashp_backcalc_kwh")
    if out.notna().sum() < min_valid:
        logger.warning(
            "Insufficient valid ASHP-on energy-balance labels (%d < %d).",
            int(out.notna().sum()),
            int(min_valid),
        )
    return out


def back_calculate_ashp_heat_hybrid(
    df: pd.DataFrame,
    *,
    energy_weight: float = 0.3,
    st_col: str = "st_kwh",
    dt_s: float = 1800.0,
) -> pd.Series:
    """Blend heuristic and energy-balance ASHP labels.

    The heuristic labels preserve realistic magnitude during charging, while the
    energy-balance labels add physical structure but can be biased low if draw,
    losses, or ST timing are imperfect. A light blend keeps the scale anchored
    to observed charging while improving consistency.
    """
    weight = float(np.clip(energy_weight, 0.0, 1.0))
    q_heur = back_calculate_ashp_heat(df, st_col=st_col, dt_s=dt_s)
    q_energy = back_calculate_ashp_heat_energy_balance(df, st_col=st_col, dt_s=dt_s)

    q_h = q_heur.to_numpy(dtype=float)
    q_e = q_energy.to_numpy(dtype=float)
    q_mix = np.full(len(df), np.nan, dtype=float)

    both = np.isfinite(q_h) & np.isfinite(q_e)
    only_h = np.isfinite(q_h) & ~np.isfinite(q_e)
    only_e = ~np.isfinite(q_h) & np.isfinite(q_e)

    q_mix[both] = (1.0 - weight) * q_h[both] + weight * q_e[both]
    q_mix[only_h] = q_h[only_h]
    q_mix[only_e] = q_e[only_e]

    return pd.Series(np.clip(q_mix, 0.0, None), index=df.index, name="Q_ashp_backcalc_kwh")


def _apply_draw_displacement(T_state: np.ndarray, f_draw: float, T_cold: float) -> np.ndarray:
    """Apply the tank's draw-displacement operator for a known fraction."""
    T_state = np.asarray(T_state, dtype=float)
    f_draw = float(np.clip(f_draw, 0.0, 1.0))

    T_new = T_state.copy()
    for i in range(3, 0, -1):
        T_new[i] = (1.0 - f_draw) * T_state[i] + f_draw * T_state[i - 1]
    T_new[0] = (1.0 - f_draw) * T_state[0] + f_draw * float(T_cold)
    return T_new


def infer_draw_off_from_temps(
    df: pd.DataFrame,
    *,
    Q_st: np.ndarray,
    Q_ashp: np.ndarray,
    Q_imm: np.ndarray,
    T_amb: np.ndarray | None = None,
    min_fraction: float = 0.02,
    max_fraction: float = 0.9,
    min_improvement: float = 0.25,
    cold_in: float | np.ndarray = 10.0,
    nominal_params: TankParams | None = None,
) -> np.ndarray:
    """Estimate draw-off volume (L) from the full 4-node profile transition.

    For each interval we first predict the next state assuming zero draw but using
    the measured heating inputs and a nominal tank model. We then solve for the
    draw-displacement fraction that best maps that no-draw state to the measured
    next profile. This uses all four nodes rather than only the top-node drop,
    which makes inferred draws more physically consistent and less likely to be
    confused with standing losses or inter-node mixing.
    """
    N = len(df)
    V_draw = np.zeros(N, dtype=float)
    T_meas = df[["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]].values

    if nominal_params is None:
        nominal_params = TankParams()

    if T_amb is None:
        T_amb_arr = df["t_amb_c"].fillna(df["t_amb_c"].median()).to_numpy(dtype=float)
    else:
        T_amb_arr = np.asarray(T_amb, dtype=float)

    if np.isscalar(cold_in):
        cold = np.full(N, float(cold_in))
    else:
        cold = np.asarray(cold_in, dtype=float)

    for k in range(N - 1):
        T_no_draw = tank_model.tank_step(
            T_meas[k],
            float(Q_st[k]),
            float(Q_ashp[k]),
            float(Q_imm[k]),
            float(T_amb_arr[k]),
            0.0,
            float(cold[k]),
            nominal_params,
        )

        draw_basis = np.array([
            float(cold[k]) - T_no_draw[0],
            T_no_draw[0] - T_no_draw[1],
            T_no_draw[1] - T_no_draw[2],
            T_no_draw[2] - T_no_draw[3],
        ])
        denom = float(np.dot(draw_basis, draw_basis))
        if denom <= 1e-9:
            continue

        f_draw = float(
            np.clip(
                np.dot(T_meas[k + 1] - T_no_draw, draw_basis) / denom,
                0.0,
                max_fraction,
            )
        )
        if f_draw <= min_fraction:
            continue

        T_with_draw = _apply_draw_displacement(T_no_draw, f_draw, float(cold[k]))
        base_err = float(np.linalg.norm(T_meas[k + 1] - T_no_draw))
        draw_err = float(np.linalg.norm(T_meas[k + 1] - T_with_draw))

        top_cooling = float(T_no_draw[3] - T_meas[k + 1, 3])
        mean_cooling = float(np.mean(T_no_draw - T_meas[k + 1]))
        if top_cooling <= 0.05 and mean_cooling <= 0.05:
            continue

        if (base_err - draw_err) <= min_improvement:
            continue

        V_draw[k] = f_draw * NODE_VOL_L

    return V_draw


def mains_temp_seasonal(
    index: pd.DatetimeIndex,
    *,
    T_mean: float = 10.5,
    T_amplitude: float = 3.5,
    peak_day_of_year: int = 244,
) -> np.ndarray:
    """Sinusoidal seasonal mains cold-water temperature [°C].

    Based on CIBSE TM65 / BS EN 806-2 typical UK ground-temperature profile:
      - annual mean ≈ 10.5 °C
      - amplitude ≈ ±3.5 °C
      - peak around day 244 (≈ 1 September)

    Parameters
    ----------
    index : DatetimeIndex
        Timestamps for the desired output.
    T_mean : float
        Annual mean mains temperature [°C].
    T_amplitude : float
        Half-range of seasonal swing [°C].
    peak_day_of_year : int
        Day of year when mains temperature peaks (default 244 = 1 Sep).

    Returns
    -------
    T_cold : np.ndarray, shape (N,)
        Estimated cold mains temperature at each timestamp.
    """
    doy = index.day_of_year.to_numpy(dtype=float)
    phase = 2.0 * np.pi * (doy - peak_day_of_year) / 365.25
    return T_mean + T_amplitude * np.cos(phase)


def prepare_inputs(df: pd.DataFrame, ashp_p: ashp_model.ASHPParams, dt_h: float = 0.5) -> dict:
    """Build arrays needed for tank simulation from a cleaned DataFrame.

    Returns dict with keys: T_meas (N,4), Q_st, Q_ashp, Q_imm, T_amb, V_draw, T_cold, T_out, P_meas (all N,).
    """
    T_meas = df[["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]].values

    # ST energy
    if "st_kwh" in df.columns:
        Q_st = df["st_kwh"].fillna(0).values
    else:
        Q_st = solar_thermal.compute_st_energy(df, dt_minutes=dt_h * 60).values

    # ASHP heat from map
    # Use lower-half tank temperatures for condenser sink proxy (coil is in lower half).
    T_sink = ashp_model.sink_proxy(df["tank_bottom_c"].values, df["tank_mid_c"].values)
    # Convert measured electrical input to delivered heat with map-based cap.
    P_meas = df["ashp_inst_kwh"].fillna(0).values
    Q_ashp = compute_ashp_heat_kwh(P_meas, df["t_out_c"].values, T_sink, ashp_p, dt_h=dt_h)

    # Immersion
    Q_imm = df["imm_tot_inst_kwh"].fillna(0).values

    T_amb = df["t_amb_c"].fillna(df["t_amb_c"].median()).values

    # Seasonal cold mains temperature — varies ~7–14 °C across the year.
    T_cold = mains_temp_seasonal(df.index)

    # Infer draw-off from the full measured profile change after accounting for
    # heating inputs and nominal passive tank dynamics.
    V_draw = infer_draw_off_from_temps(
        df,
        Q_st=Q_st,
        Q_ashp=Q_ashp,
        Q_imm=Q_imm,
        T_amb=T_amb,
        min_fraction=0.02,
        max_fraction=0.9,
        min_improvement=0.25,
        cold_in=T_cold,
        nominal_params=TankParams(),
    )

    T_out = df["t_out_c"].values

    return dict(
        T_meas=T_meas,
        Q_st=Q_st,
        Q_ashp=Q_ashp,
        Q_imm=Q_imm,
        T_amb=T_amb,
        V_draw=V_draw,
        T_cold=T_cold,
        T_out=T_out,
        P_meas=P_meas,
    )


def fit_tank_params(
    inputs: dict,
    *,
    max_nfev: int = 300,
    reg_weight: float = 0.2,
    rollout_weight: float = 0.30,
    rollout_horizon: int = 48,
    rollout_stride: int = 96,
    mains_mean_offset_init: float = 0.0,
    mains_mean_offset_bounds: tuple[float, float] = (-3.0, 3.0),
    mains_mean_offset_reg_weight: float = 0.5,
    ashp_p: "ashp_model.ASHPParams | None" = None,
) -> tank_model.TankParams:
    """Fit tank parameters using one-step-ahead (teacher-forced) residuals.

    Each step resets to the measured state, so the residuals are the
    one-step prediction errors.  This avoids error accumulation and gives
    stable parameter estimates.

    If ``ashp_p`` is provided the multi-step rollout sections recompute
    Q_ashp from the rolling predicted tank state (closed-loop coupling),
    so the optimiser is penalised for parameters that cause autonomous drift
    rather than only for single-step errors.
    """
    T_meas = inputs["T_meas"]
    Q_st   = inputs["Q_st"]
    Q_ashp = inputs["Q_ashp"]
    Q_imm  = inputs["Q_imm"]
    T_amb  = inputs["T_amb"]
    V_draw = inputs["V_draw"]
    T_cold = inputs["T_cold"]
    # T_out and P_meas are only used when ashp_p is supplied for CL rollout.
    T_out_arr  = inputs.get("T_out",  np.zeros(len(Q_st)))
    P_meas_arr = inputs.get("P_meas", np.zeros(len(Q_st)))
    N = len(Q_st)
    steps = N - 1

    p0 = tank_model.TankParams()
    p0_vec = p0.to_vector()
    lb_tank = tank_model.TankParams.lower_bounds()
    ub_tank = tank_model.TankParams.upper_bounds()
    scale_tank = np.maximum(ub_tank - lb_tank, 1e-9)

    # Fit a global mains-temperature mean offset (delta around seasonal curve)
    # jointly with tank parameters to absorb site-specific seasonal bias.
    mains_lb, mains_ub = mains_mean_offset_bounds
    x0 = np.concatenate([p0_vec, [mains_mean_offset_init]])
    lb = np.concatenate([lb_tank, [mains_lb]])
    ub = np.concatenate([ub_tank, [mains_ub]])
    scale = np.maximum(ub - lb, 1e-9)

    # Clamp x0 within bounds
    x0 = np.clip(x0, lb + 1e-8, ub - 1e-8)

    logger.info(
        "Starting tank fit (max_nfev=%d, rollout_weight=%.3f, horizon=%d, stride=%d). "
        "This stage can take several minutes.",
        max_nfev,
        rollout_weight,
        rollout_horizon,
        rollout_stride,
    )
    t0 = time.perf_counter()
    eval_count = 0
    # Estimated total residual evals: scipy uses forward finite differences
    # for the Jacobian → (n_params + 1) residual calls per outer iteration.
    _n_params = len(x0)
    _eta_total_evals = max_nfev * (_n_params + 1)

    def residuals(x):
        nonlocal eval_count
        eval_count += 1
        elapsed = time.perf_counter() - t0
        if eval_count == 1:
            eta_s = elapsed * _eta_total_evals
            logger.info(
                "Tank fit progress: residual eval 1, elapsed %.2f s — "
                "estimated total ~%.0f s (~%.1f min) for max_nfev=%d",
                elapsed, eta_s, eta_s / 60.0, max_nfev,
            )
        elif eval_count % 10 == 0:
            rate = elapsed / eval_count
            remaining = max(0, _eta_total_evals - eval_count) * rate
            logger.info(
                "Tank fit progress: residual eval %d / ~%d, elapsed %.0f s, ETA ~%.0f s",
                eval_count, _eta_total_evals, elapsed, remaining,
            )

        tank_x = x[:-1]
        mains_offset = float(x[-1])
        p = tank_model.TankParams.from_vector(tank_x)
        T_cold_eff = T_cold + mains_offset
        # One-step-ahead: ALL steps in a single vectorised call (no Python loop).
        T_pred = tank_model.tank_step_batch(
            T_meas[:steps],
            Q_st[:steps], Q_ashp[:steps], Q_imm[:steps],
            T_amb[:steps], V_draw[:steps], T_cold_eff[:steps],
            p,
        )
        err_1step = (T_pred - T_meas[1: steps + 1]).ravel()

        # Multi-step rollout residuals to reduce autonomous drift.
        # When ashp_p is provided, Q_ashp is recomputed each sub-step from
        # the rolling predicted state (closed-loop), matching how the model
        # will actually be evaluated.  Otherwise the teacher-forced Q_ashp
        # from prepare_inputs is used (faster but mis-aligned with CL eval).
        err_rollout_parts = []
        if rollout_weight > 0 and rollout_horizon > 1:
            max_start = max(0, len(Q_st) - rollout_horizon)
            for s in range(0, max_start + 1, max(1, rollout_stride)):
                T_roll = T_meas[s].copy()
                for j in range(rollout_horizon):
                    k = s + j
                    if ashp_p is not None:
                        # Closed-loop: derive Q_ashp from predicted T_sink
                        T_sink_roll = ashp_model.sink_proxy(T_roll[0], T_roll[1])
                        Q_ashp_k = float(
                            compute_ashp_heat_kwh(
                                P_meas_arr[k],
                                T_out_arr[k],
                                T_sink_roll,
                                ashp_p,
                            )
                        )
                    else:
                        Q_ashp_k = float(Q_ashp[k])
                    T_roll = tank_model.tank_step(
                        T_roll,
                        float(Q_st[k]),
                        Q_ashp_k,
                        float(Q_imm[k]),
                        float(T_amb[k]),
                        float(V_draw[k]),
                        float(T_cold_eff[k]),
                        p,
                    )
                    err_rollout_parts.append(T_roll - T_meas[k + 1])

        if err_rollout_parts:
            err_rollout = np.asarray(err_rollout_parts, dtype=float).ravel()
            err = np.concatenate([err_1step, rollout_weight * err_rollout])
        else:
            err = err_1step

        # Dimensionless regularisation toward physically informed defaults.
        # Scale by sqrt(n_res) so priors remain influential for long datasets.
        reg_scale = np.sqrt(max(1, err.size))
        reg_tank = reg_weight * reg_scale * ((tank_x - p0_vec) / scale_tank)
        mains_scale = max(1e-9, mains_ub - mains_lb)
        reg_mains = np.array([
            mains_mean_offset_reg_weight * reg_scale * (mains_offset / mains_scale)
        ])
        reg = np.concatenate([reg_tank, reg_mains])
        return np.concatenate([err, reg])

    result = least_squares(
        residuals, x0,
        bounds=(lb, ub),
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=max_nfev,
        verbose=0,
    )
    logger.info(
        "Tank fit completed: cost=%.2f, nfev=%d, residual_evals=%d, elapsed=%.1f s",
        result.cost,
        result.nfev,
        eval_count,
        time.perf_counter() - t0,
    )
    return tank_model.TankParams.from_vector(result.x)


def simulate_closed_loop(
    T0: np.ndarray,
    Q_st: np.ndarray,
    Q_imm: np.ndarray,
    T_amb: np.ndarray,
    V_draw: np.ndarray,
    T_cold: np.ndarray,
    T_out: np.ndarray,
    P_meas: np.ndarray,
    ashp_p: ashp_model.ASHPParams,
    params: tank_model.TankParams,
    dt_s: float = 1800.0,
) -> np.ndarray:
    """Simulate the tank autonomously (closed-loop for ASHP as well).

    Uses predicted tank state to compute T_sink, then Q_ashp from ASHP map.
    """
    N = len(Q_st)
    T_hist = np.zeros((N + 1, 4))
    T_hist[0] = T0

    for k in range(N):
        # Compute T_sink from current predicted state
        T_sink = ashp_model.sink_proxy(T_hist[k, 0], T_hist[k, 1])  # bottom and mid
        # Compute Q_ashp using predicted T_sink with a capacity cap.
        Q_ashp_k = float(compute_ashp_heat_kwh(P_meas[k], T_out[k], T_sink, ashp_p))

        T_hist[k + 1] = tank_model.tank_step(
            T_hist[k],
            float(Q_st[k]),
            float(Q_ashp_k),
            float(Q_imm[k]),
            float(T_amb[k]),
            float(V_draw[k]),
            float(T_cold[k]),
            params,
            dt_s,
        )
    return T_hist


def simulate_closed_loop_with_diagnostics(
    T0: np.ndarray,
    Q_st: np.ndarray,
    Q_imm: np.ndarray,
    T_amb: np.ndarray,
    V_draw: np.ndarray,
    T_cold: np.ndarray,
    T_out: np.ndarray,
    P_meas: np.ndarray,
    ashp_p: ashp_model.ASHPParams,
    params: tank_model.TankParams,
    dt_s: float = 1800.0,
    report_steps: int = 24,
) -> tuple[np.ndarray, list[dict]]:
    """Simulate closed-loop and record diagnostic terms for first report_steps."""
    N = len(Q_st)
    T_hist = np.zeros((N + 1, 4))
    T_hist[0] = T0
    diagnostics = []

    for k in range(N):
        T_sink = ashp_model.sink_proxy(T_hist[k, 0], T_hist[k, 1])
        cop = ashp_model.predict_cop(T_out[k], T_sink, ashp_p)
        Q_ashp_k = float(compute_ashp_heat_kwh(P_meas[k], T_out[k], T_sink, ashp_p))

        if k < report_steps:
            diagnostics.append({
                "k": int(k),
                "T_prev": T_hist[k].tolist(),
                "T_sink": float(T_sink),
                "T_out": float(T_out[k]),
                "P_meas": float(P_meas[k]),
                "COP": float(cop),
                "Q_ashp": float(Q_ashp_k),
            })

        T_hist[k + 1] = tank_model.tank_step(
            T_hist[k],
            float(Q_st[k]),
            float(Q_ashp_k),
            float(Q_imm[k]),
            float(T_amb[k]),
            float(V_draw[k]),
            float(T_cold[k]),
            params,
            dt_s,
        )

    return T_hist, diagnostics


def simulate_closed_loop_with_energy_audit(
    T0: np.ndarray,
    Q_st: np.ndarray,
    Q_imm: np.ndarray,
    T_amb: np.ndarray,
    V_draw: np.ndarray,
    T_cold: np.ndarray,
    T_out: np.ndarray,
    P_meas: np.ndarray,
    ashp_p: ashp_model.ASHPParams,
    params: tank_model.TankParams,
    dt_s: float = 1800.0,
    report_steps: int = 24,
) -> tuple[np.ndarray, list[dict]]:
    """Simulate closed-loop and capture full tank energy-budget terms.

    Each audit row includes ASHP coupling terms and per-node tank terms in kJ.
    """
    N = len(Q_st)
    T_hist = np.zeros((N + 1, 4))
    T_hist[0] = T0
    audits = []

    for k in range(N):
        T_sink = ashp_model.sink_proxy(T_hist[k, 0], T_hist[k, 1])
        cop = ashp_model.predict_cop(T_out[k], T_sink, ashp_p)
        Q_ashp_k = float(compute_ashp_heat_kwh(P_meas[k], T_out[k], T_sink, ashp_p))

        T_next, breakdown = tank_model.tank_step_with_breakdown(
            T_hist[k],
            float(Q_st[k]),
            Q_ashp_k,
            float(Q_imm[k]),
            float(T_amb[k]),
            float(V_draw[k]),
            float(T_cold[k]),
            params,
            dt_s,
        )
        T_hist[k + 1] = T_next

        if k < report_steps:
            row = {
                "k": int(k),
                "T_sink": float(T_sink),
                "T_out": float(T_out[k]),
                "P_meas_kwh": float(P_meas[k]),
                "COP": float(cop),
                "Q_ashp_kwh": Q_ashp_k,
            }
            row.update(breakdown)
            audits.append(row)

    return T_hist, audits


def run_identification(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.7,
    max_nfev: int = 300,
    fit_tank: bool = True,
    tank_fit_kwargs: dict | None = None,
    fixed_tank_params: tank_model.TankParams | None = None,
) -> tuple[IdentificationResult, pd.DataFrame, pd.DataFrame]:
    """Full identification pipeline.

    Returns
    -------
    result : IdentificationResult
    df_train : training slice
    df_val : validation slice
    """
    # Compute ST energy column
    df = df.copy()
    df["st_kwh"] = solar_thermal.compute_st_energy(df)

    # Train/val split by time
    split_idx = int(len(df) * train_frac)
    df_train = df.iloc[:split_idx].copy()
    df_val   = df.iloc[split_idx:].copy()

    logger.info("Train: %d rows, Val: %d rows", len(df_train), len(df_val))

    # Step 1 (Pass 1): Fit ASHP power (b) coefficients only
    T_sink_train = ashp_model.sink_proxy(
        df_train["tank_bottom_c"].values,
        df_train["tank_mid_c"].values,
    )
    ashp_p = ashp_model.fit_ashp_maps(
        T_out=df_train["t_out_c"].values,
        T_sink=T_sink_train,
        Q_meas_kwh=None,
        P_meas_kwh=df_train["ashp_inst_kwh"].values,
    )

    # Step 1 (Pass 2): Back-calculate heat delivery and re-fit both a and b
    Q_back = back_calculate_ashp_heat_hybrid(df_train, energy_weight=0.3)
    df_train["Q_ashp_backcalc_kwh"] = Q_back
    df_val["Q_ashp_backcalc_kwh"] = np.nan

    if Q_back.notna().sum() >= 50:
        ashp_p = ashp_model.fit_ashp_maps(
            T_out=df_train["t_out_c"].values,
            T_sink=T_sink_train,
            Q_meas_kwh=Q_back.values,
            P_meas_kwh=df_train["ashp_inst_kwh"].values,
        )
        logger.info("ASHP capacity fitted from back-calculated heat data.")
    else:
        logger.warning("Insufficient ASHP-only intervals for back-calculation; "
                       "fallback to a = b * 3.0 retained.")

    # Step 2: Fit tank on training data unless we are iterating on ASHP labels
    # and want a cheap run that reuses a fixed tank parameter set.
    if fit_tank:
        train_inputs = prepare_inputs(df_train, ashp_p)
        tank_fit_options = dict(tank_fit_kwargs or {})
        tank_fit_options.setdefault("max_nfev", max_nfev)
        tank_fit_options.setdefault("ashp_p", ashp_p)
        tank_p = fit_tank_params(train_inputs, **tank_fit_options)
    else:
        tank_p = fixed_tank_params or tank_model.TankParams()
        logger.info(
            "Skipping tank fit for fast ASHP iteration; using %s tank parameters.",
            "provided" if fixed_tank_params is not None else "default",
        )

    result = IdentificationResult(
        tank_params=tank_p,
        ashp_params=ashp_p,
        hx_effectiveness=1.0,
        cost_history=[],
    )
    return result, df_train, df_val
