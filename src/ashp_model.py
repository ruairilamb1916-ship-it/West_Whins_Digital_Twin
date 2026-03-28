"""
ASHP performance maps (control-oriented).

A performance map is a compact mathematical equation that describes how the heat pump
behaves without simulating its internal physics.

Outputs: Q_cond (condenser heat output) and P_elec (electrical power input).
Inputs: T_out (outdoor air temperature) and T_sink (tank sink-proxy temperature).  The sink-proxy is a weighted average
of the mid and top node temperatures, representing the effective sink temperature seen by the ASHP.

Compact parametric relationships for capacity and electrical power
as functions of ambient temperature and tank sink-proxy temperature.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)

# This percentile is used later for filtering
# This ensures the map is fitted to steady full-load operation, not partial or start-up which would distort results.
HIGH_LOAD_PERCENTILE = 75
COP_INTERCEPT_OFFSET = -0.1
STRATIFICATION_SINK_GAIN = 0.08
STRATIFICATION_SINK_MAX_C = 20.0


@dataclass
class ASHPParams:
    """Identified ASHP map parameters.
    
    Assume a bilinear form for both capacity and power maps:

    Map input temperature is outdoor air temperature (T_out).

    Capacity  Q̇_cond = a0 + a1·T_out + a2·T_sink + a3·T_out·T_sink
    Power     P_elec = b0 + b1·T_out + b2·T_sink + b3·T_out·T_sink
    COP       COP    = c0 + c1·T_out + c2·T_sink + c3·T_out·T_sink + c4·T_out^2 + c5·T_sink^2
    """
    a: np.ndarray = field(default_factory=lambda: np.array([8.0, 0.1, -0.05, 0.0]))
    b: np.ndarray = field(default_factory=lambda: np.array([3.0, -0.02, 0.03, 0.0]))
    c: np.ndarray = field(default_factory=lambda: np.array([3.0, 0.02, -0.02, 0.0, 0.0, 0.0]))

    def to_dict(self) -> dict[str, list[float]]:
        """Serialize map coefficients for persistence."""
        return {
            "a": np.asarray(self.a, dtype=float).tolist(),
            "b": np.asarray(self.b, dtype=float).tolist(),
            "c": np.asarray(self.c, dtype=float).tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ASHPParams":
        """Deserialize coefficients; accept legacy payloads without COP map."""
        base = cls()
        a = np.asarray(payload.get("a", base.a), dtype=float)
        b = np.asarray(payload.get("b", base.b), dtype=float)
        c = np.asarray(payload.get("c", base.c), dtype=float)
        return cls(a=a, b=b, c=c)


def sink_proxy(
    T_mid: np.ndarray,
    T_top: np.ndarray,
    w_mid: float = 0.5,
    w_top: float = 0.5,
) -> np.ndarray:
    """Weighted average of mid and top node temperatures."""
    return w_mid * np.asarray(T_mid) + w_top * np.asarray(T_top)


def effective_sink_temperature(
    T_sink: np.ndarray | float,
    T_bottom: np.ndarray | float | None = None,
    T_top: np.ndarray | float | None = None,
) -> np.ndarray:
    """Apply a bounded stratification correction to the sink temperature."""
    T_s = np.asarray(T_sink, dtype=float)
    if T_bottom is None or T_top is None:
        return T_s

    spread = np.asarray(T_top, dtype=float) - np.asarray(T_bottom, dtype=float)
    spread = np.clip(spread, 0.0, STRATIFICATION_SINK_MAX_C)
    return T_s + STRATIFICATION_SINK_GAIN * spread


def predict_capacity(T_out: np.ndarray, T_sink: np.ndarray, p: ASHPParams) -> np.ndarray:
    """Predict condenser heat output [kW]."""
    a = p.a
    T_a, T_s = np.asarray(T_out, dtype=float), np.asarray(T_sink, dtype=float)
    return np.maximum(a[0] + a[1] * T_a + a[2] * T_s + a[3] * T_a * T_s, 0.0)


def predict_power(T_out: np.ndarray, T_sink: np.ndarray, p: ASHPParams) -> np.ndarray:
    """Predict electrical power [kW]."""
    b = p.b
    T_a, T_s = np.asarray(T_out, dtype=float), np.asarray(T_sink, dtype=float)
    return np.maximum(b[0] + b[1] * T_a + b[2] * T_s + b[3] * T_a * T_s, 0.1)


def predict_cop(T_out: np.ndarray, T_sink: np.ndarray, p: ASHPParams) -> np.ndarray:
    """Predict COP from a direct fitted map."""
    c = p.c
    T_a, T_s = np.asarray(T_out, dtype=float), np.asarray(T_sink, dtype=float)
    return np.clip(
        (c[0] + COP_INTERCEPT_OFFSET) + c[1] * T_a + c[2] * T_s + c[3] * (T_a ** 2) + c[4] * (T_s ** 2) + c[5] * T_a * T_s,
        1.0,
        4.0,
    )


def fit_ashp_maps(
    T_out: np.ndarray,
    T_sink: np.ndarray,
    Q_meas_kwh: np.ndarray,
    P_meas_kwh: np.ndarray,
    Q_cop_meas_kwh: np.ndarray | None = None,
    dt_h: float = 0.5,
) -> ASHPParams:
    """Fit ASHP capacity/power maps and a direct COP map from measured interval energies.
    Stage 1: Data Filtering
    Stage 2: Power Map Fitting
    Stage 3: COP Map Fitting

    Parameters
    ----------
    T_out, T_sink : outdoor air temperature and sink-proxy arrays [°C].
    Q_meas_kwh : measured condenser heat per interval [kWh] (may be unknown;
                 pass NaN to skip COP fitting – power only).
    P_meas_kwh : measured ASHP electrical energy per interval [kWh].
    dt_h : interval length in hours (default 0.5).

    Returns
    -------
    ASHPParams with fitted coefficients.
    """
    T_a = np.asarray(T_out, dtype=float)
    T_s = np.asarray(T_sink, dtype=float)
    P_meas = np.asarray(P_meas_kwh, dtype=float)

    # Mask: only intervals where ASHP was running at substantial load.
    # Use a high percentile threshold so we fit steady-state power
    # (not partial duty-cycle intervals).
    valid = np.isfinite(P_meas) & (P_meas > 0.05) & np.isfinite(T_a) & np.isfinite(T_s) # Removes intervals with NaN Temps or near-zero ASHP power
    # If there are enough intervals (>50) we can apply high-load filter (>75%)
    if valid.sum() > 50:
        p75 = np.percentile(P_meas[valid], HIGH_LOAD_PERCENTILE)
        mask = valid & (P_meas >= p75)
    else:
        mask = valid
    T_a_f, T_s_f, P_f = T_a[mask], T_s[mask], P_meas[mask]

    # Convert interval kWh → average kW
    P_kw = P_f / dt_h

    # Fit power map: P_elec = b0 + b1*T_a + b2*T_s + b3*T_a*T_s
    X = np.column_stack([np.ones(len(T_a_f)), T_a_f, T_s_f, T_a_f * T_s_f])

    # Use Ordinary Least Squares (OLS) for a good initial guess, then refine with robust loss
    # OLS finds the b that minimises the sum of squared errors between X @ b (@ is matrix multiplication).
    b_ols, _, _, _ = np.linalg.lstsq(X, P_kw, rcond=None)
    b_lo = np.array([-20.0, -0.5, -0.5, -0.02])
    b_hi = np.array([20.0,   0.5,  0.5,  0.02])
    b_init = np.clip(b_ols, b_lo + 1e-6, b_hi - 1e-6) # ensure initial guess is within bounds for least_squares

    def power_residuals(b):
        pred = X @ b
        pred = np.maximum(pred, 0.1) 
        return pred - P_kw

    # 
    res_b = least_squares(
        power_residuals, b_init,
        bounds=(b_lo, b_hi), # constrain to physically plausible values
        loss="soft_l1", # less sensitive to outliers than squared loss
    )

    params = ASHPParams()
    params.b = res_b.x
    logger.info("ASHP power map coefficients: %s", params.b)

    # Capacity fit: if Q_meas available
    Q_meas = np.asarray(Q_meas_kwh, dtype=float) if Q_meas_kwh is not None else np.full_like(P_meas, np.nan)
    mask_q = mask & np.isfinite(Q_meas) & (Q_meas > 0.01)

    if mask_q.sum() > 20:
        T_a_q, T_s_q = T_a[mask_q], T_s[mask_q]
        Q_kw = Q_meas[mask_q] / dt_h
        Xq = np.column_stack([np.ones(len(T_a_q)), T_a_q, T_s_q, T_a_q * T_s_q])
        a_init = np.array([8.0, 0.1, -0.05, 0.0])

        def cap_residuals(a):
            pred = Xq @ a
            pred = np.maximum(pred, 0.0)
            return pred - Q_kw

        logger.info("No. of valid data points used for capacity fitting: %d", mask_q.sum())
        res_a = least_squares(cap_residuals, a_init, loss="soft_l1")
        params.a = res_a.x
        logger.info("ASHP capacity map coefficients: %s", params.a)
    else:
        logger.warning(
            "Insufficient valid ASHP heat labels for capacity fitting (%d <= 20); "
            "skipping capacity fit.",
            mask_q.sum(),
        )

    # Direct COP fit from provided COP labels (defaults to Q_meas) and measured electrical input.
    # Capacity (a) and power (b) fitting above are unchanged; weighting is applied only in this COP fit.
    Q_cop_meas = (
        np.asarray(Q_cop_meas_kwh, dtype=float)
        if Q_cop_meas_kwh is not None
        else Q_meas
    )
    mask_cop = (
        np.isfinite(Q_cop_meas)
        & (Q_cop_meas > 0.01)
        & np.isfinite(P_meas)
        & (P_meas > 0.05)
        & np.isfinite(T_a)
        & np.isfinite(T_s)
    )

    if mask_cop.sum() > 20:
        T_a_c, T_s_c = T_a[mask_cop], T_s[mask_cop]
        COP_meas = Q_cop_meas[mask_cop] / P_meas[mask_cop]
        # Clip COP targets before fitting to stabilise training against extreme label noise.
        COP_meas = np.clip(COP_meas, 1.0, 4.0)

        # Upweight higher-COP (lower-lift, warmer) operation so the COP map better tracks that regime.
        # Construct weights from measured COP, clipped to [1, 4] to limit leverage of extreme points.
        weights = np.clip(COP_meas, 1.0, 4.0)
        # Use stronger power weighting to emphasize warmer/high-COP regimes.
        weights = (weights ** 1.5) / np.mean(weights ** 1.5)

        Xc = np.column_stack([
            np.ones(len(T_a_c)),
            T_a_c,
            T_s_c,
            T_a_c ** 2,
            T_s_c ** 2,
            T_a_c * T_s_c,
        ])
        c_init = np.array([3.0, 0.02, -0.02, 0.0, 0.0, 0.0])

        def cop_residuals(c):
            pred = Xc @ c
            pred = np.clip(pred, 1.0, 8.0)
            # Weighted residuals plus light L2 regularisation improve warm-period fit while keeping coefficients stable.
            residuals = weights * (pred - COP_meas)

            lambda_reg = 0.001
            residuals = np.concatenate([
                residuals,
                np.sqrt(lambda_reg) * c
            ])
            return residuals

        logger.info("No. of valid data points used for COP fitting: %d", mask_cop.sum())
        res_c = least_squares(cop_residuals, c_init, loss="soft_l1")
        c = res_c.x
        params.c = c
        logger.info("ASHP COP map coefficients: %s", params.c)
        print(
            "Fitted COP coefficients: "
            f"c0={c[0]:.6f}, c1={c[1]:.6f}, c2={c[2]:.6f}, "
            f"c3={c[3]:.6f}, c4={c[4]:.6f}, c5={c[5]:.6f}"
        )
        print("COP coeffs:", c)
    else:
        logger.warning(
            "Insufficient valid ASHP rows for COP fitting (%d <= 20); "
            "skipping COP fit.",
            mask_cop.sum(),
        )

    return params
