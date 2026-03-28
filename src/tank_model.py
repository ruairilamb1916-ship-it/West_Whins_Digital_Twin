"""
4-node DHW tank grey-box model.

States:  T_b, T_m, T_mh, T_t  (bottom, mid, mid-hi, top).
Inputs per interval:
  - Q_ST   : solar-thermal heat delivered [kWh]
  - Q_ASHP : ASHP condenser heat delivered [kWh]
  - Q_imm  : immersion heater heat [kWh]
  - T_amb  : ambient (plant room) temperature [°C]

The tank is 550 L split into 4 equal-volume nodes (137.5 L each).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Physical constants
RHO = 1000.0   # kg/m³
CP  = 4.186     # kJ/(kg·K)
NODE_VOL_L = 550.0 / 4.0   # litres per node
NODE_MASS = NODE_VOL_L * RHO / 1000.0   # kg  (137.5 kg)
NODE_CAP  = NODE_MASS * CP  # kJ/K  (≈575.3)
DHW_EXTRACTION_FRACTIONS = np.array([0.0, 0.05, 0.25, 0.70], dtype=float)


def _soft_enforce_stratification_row(T_row: np.ndarray) -> np.ndarray:
    """Apply minimal local averaging to enforce bottom<=mid<=mid_hi<=top.

    This is a numerically stable post-step correction that removes node inversions
    without introducing abrupt global mixing.
    """
    x = np.asarray(T_row, dtype=float).copy()
    # A few forward sweeps are sufficient for 4 nodes.
    for _ in range(6):
        changed = False
        if x[0] > x[1]:
            m = 0.5 * (x[0] + x[1])
            x[0] = m
            x[1] = m
            changed = True
        if x[1] > x[2]:
            m = 0.5 * (x[1] + x[2])
            x[1] = m
            x[2] = m
            changed = True
        if x[2] > x[3]:
            m = 0.5 * (x[2] + x[3])
            x[2] = m
            x[3] = m
            changed = True
        if not changed:
            break
    return x


def _soft_enforce_stratification_batch(T_mat: np.ndarray) -> np.ndarray:
    """Vector helper for row-wise stratification enforcement."""
    T_out = np.asarray(T_mat, dtype=float).copy()
    for n in range(T_out.shape[0]):
        T_out[n] = _soft_enforce_stratification_row(T_out[n])
    return T_out


def _st_acceptance_factor(T_hot: float) -> float:
        """Fraction of ST heat accepted as upper-layer tank temperature rises.

        The effective charging path curtails strongly when any upper node is hot,
        not only when the top sensor is high. Use a tighter linear taper:
            - full acceptance at <= 60 C
            - zero acceptance at >= 70 C
        """
        return float(np.clip((70.0 - float(T_hot)) / 10.0, 0.0, 1.0))


def _ashp_acceptance_factor(T_hot: float) -> float:
        """Fraction of ASHP heat accepted by DHW tank at high temperatures.

        Models thermostat/setpoint behavior where DHW charging is curtailed as the
        cylinder approaches target temperature.
            - full acceptance at <= 53 C
            - zero acceptance at >= 63 C
        """
        return float(np.clip((63.0 - float(T_hot)) / 10.0, 0.0, 1.0))


def _normalise_fractions(f: np.ndarray) -> np.ndarray:
    """Return non-negative fractions that sum to 1.

    This prevents optimisation from collapsing heat-split vectors toward zero,
    which would otherwise suppress all injected energy in closed-loop rollout.
    """
    f = np.asarray(f, dtype=float)
    f = np.clip(f, 0.0, None)
    s = float(np.sum(f))
    if s <= 1e-12:
        return np.full_like(f, 1.0 / len(f), dtype=float)
    return f / s


@dataclass
class TankParams:
    """Contains all the learnable parameters of the grey-box model.
    These are the parameters that will be optimised to fit the model to real data.

    UA_loss : per-node UA to ambient [kW/K] (4 values, bottom→top).
    UA_adj  : adjacent-node conductance [kW/K] (3 values: b-m, m-mh, mh-t).
    f_st    : fraction of ST heat to each node (4 values, should sum ≈1).
    f_ashp  : fraction of ASHP heat to each node (4 values).
    f_imm   : fraction of immersion heat to each node (4 values).
    mix_coeff : draw-induced mixing coefficient [kW/K].
    alpha_draw : draw-off effectiveness factor [dimensionless].
    T_mains : cold mains water temperature [°C].
    """
    #default values are physically informed intitial guesses
    UA_loss: np.ndarray = field(default_factory=lambda: np.array([0.003, 0.002, 0.002, 0.003]))
    # Inter-node conductance: pure water conduction ≈ 0.001 kW/K; allow up to
    # ~5× for mild buoyancy-driven mixing.  Prior centred on a small value.
    UA_adj:  np.ndarray = field(default_factory=lambda: np.array([0.003, 0.003, 0.003]))
    f_st:    np.ndarray = field(default_factory=lambda: np.array([0.0, 0.3, 0.5, 0.2]))
    # ASHP condenser coil typically sits in the lower half of the tank.
    f_ashp:  np.ndarray = field(default_factory=lambda: np.array([0.4, 0.4, 0.15, 0.05]))
    f_imm:   np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.2, 0.8]))
    mix_coeff: float = 0.01
    alpha_draw: float = 1.0
    T_mains: float = 10.0  # cold mains water temperature [°C]

    def to_vector(self) -> np.ndarray:
        """Flatten all parameters to a 1-D vector for optimisation."""
        return np.concatenate([
            self.UA_loss,       # 4
            self.UA_adj,        # 3
            self.f_st,          # 4
            self.f_ashp,        # 4
            self.f_imm,         # 4
            [self.mix_coeff],   # 1
            [self.alpha_draw],  # 1
        ])                      # total = 21

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "TankParams":
        """Reverses to_vector, reconstructs a TankParams instance with array slicing"""
        p = cls()
        p.UA_loss    = v[0:4]
        p.UA_adj     = v[4:7]
        p.f_st       = v[7:11]
        p.f_ashp     = v[11:15]
        p.f_imm      = v[15:19]
        p.mix_coeff  = float(v[19])
        p.alpha_draw = float(v[20])
        return p

    @staticmethod
    def lower_bounds() -> np.ndarray:
        return np.array([
            0, 0, 0, 0,               # UA_loss
            0, 0, 0,                   # UA_adj
            0, 0, 0, 0,               # f_st
            0, 0, 0, 0,               # f_ashp
            0, 0, 0, 0,               # f_imm
            0,                         # mix_coeff
            0,                         # alpha_draw
        ], dtype=float)

    @staticmethod
    def upper_bounds() -> np.ndarray:
        return np.array([
            # UA_loss: physically, a 550 L cylinder loses ≤ ~5 kWh/day in standby.
            # At ΔT=45 K that implies UA_total ≤ 0.0046 kW/K.
            # Cap each node at 0.005 kW/K so the total stays within physical range
            # and the optimiser cannot substitute unmodelled draw losses with
            # spuriously high wall losses, which collapses the closed-loop energy budget.
            0.005, 0.005, 0.005, 0.005,   # UA_loss [kW/K]
            # UA_adj: pure water conduction ≈ 0.001 kW/K; cap at 0.012 kW/K
            # (~20× conduction) to allow mild convective mixing without letting
            # the optimiser destroy stratification to fit one-step residuals.
            0.012, 0.012, 0.012,          # UA_adj  [kW/K]
            1, 1, 1, 1,                   # f_st
            1, 1, 1, 1,                   # f_ashp
            1, 1, 1, 1,                   # f_imm
            0.2,                           # mix_coeff
            2.0,                           # alpha_draw
        ], dtype=float)


def tank_step_batch(
    T: np.ndarray,
    Q_st_kwh: np.ndarray,
    Q_ashp_kwh: np.ndarray,
    Q_imm_kwh: np.ndarray,
    T_amb: np.ndarray,
    V_draw_l: np.ndarray,
    T_cold: np.ndarray | float,
    params: "TankParams",
    dt_s: float = 1800.0,
    Q_dhw_kwh: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorised tank step for N *independent* steps (one-step-ahead use only).

    Parameters
    ----------
    T          : (N, 4) current temperatures [°C].
    Q_st_kwh   : (N,) solar-thermal input [kWh].
    Q_ashp_kwh : (N,) ASHP input [kWh].
    Q_imm_kwh  : (N,) immersion input [kWh].
    T_amb      : (N,) ambient temperature [°C].
    V_draw_l   : (N,) draw-off volume [L].
    T_cold     : scalar or (N,) cold mains temperature [°C].
    params     : TankParams instance.
    dt_s       : timestep in seconds.

    Returns
    -------
    T_new : (N, 4) updated temperatures [°C].
    """
    T = np.asarray(T, dtype=float)            # (N, 4)
    Q_st   = np.asarray(Q_st_kwh,   dtype=float) * 3600.0  # (N,) kJ
    Q_ashp = np.asarray(Q_ashp_kwh, dtype=float) * 3600.0
    Q_imm  = np.asarray(Q_imm_kwh,  dtype=float) * 3600.0
    if Q_dhw_kwh is None:
        Q_dhw = np.zeros(len(T), dtype=float)
    else:
        Q_dhw = np.asarray(Q_dhw_kwh, dtype=float) * 3600.0

    # Use hottest upper-layer node (mid, mid-hi, top) as acceptance proxy.
    T_hot = np.max(T[:, 1:], axis=1)
    Q_st   *= np.clip((70.0 - T_hot) / 10.0, 0.0, 1.0)
    Q_ashp *= np.clip((63.0 - T_hot) / 10.0, 0.0, 1.0)

    f_st   = _normalise_fractions(params.f_st)    # (4,)
    f_ashp = _normalise_fractions(params.f_ashp)
    f_imm  = _normalise_fractions(params.f_imm)

    # Step 1: heat injection — broadcasting (N,1)×(1,4) → (N,4)
    dQ = (
        Q_st[:, None]   * f_st[None, :]
        + Q_ashp[:, None] * f_ashp[None, :]
        + Q_imm[:, None]  * f_imm[None, :]
    )
    T_temp = T + dQ / NODE_CAP

    # Step 2: losses, inter-node conduction, mixing
    T_amb_nd = np.asarray(T_amb, dtype=float)
    UA_loss = params.UA_loss   # (4,)
    UA_adj  = params.UA_adj    # (3,)
    mc      = params.mix_coeff

    loss = UA_loss[None, :] * (T_temp - T_amb_nd[:, None]) * dt_s  # (N,4)

    # diff_up[n,j] = T_temp[n,j+1] - T_temp[n,j] for j=0,1,2  (contribution from above)
    # diff_dn[n,j] = T_temp[n,j]   - T_temp[n,j+1]              (contribution from below to j+1)
    diff_up = (T_temp[:, 1:] - T_temp[:, :-1]) * dt_s   # (N,3)
    diff_dn = -diff_up                                    # (N,3)

    cond = np.zeros_like(T_temp)
    cond[:, :-1] += UA_adj[None, :] * diff_up   # nodes 0-2 gain from node above
    cond[:, 1:]  += UA_adj[None, :] * diff_dn   # nodes 1-3 gain from node below

    mix = np.zeros_like(T_temp)
    mix[:, :-1] += mc * diff_up
    mix[:, 1:]  += mc * diff_dn

    T_temp2 = T_temp + (-loss + cond + mix) / NODE_CAP

    # Step 2b: explicit DHW heat extraction, applied preferentially to upper nodes.
    T_temp2 = T_temp2 - (Q_dhw[:, None] * DHW_EXTRACTION_FRACTIONS[None, :]) / NODE_CAP

    # Step 3: draw-off displacement (cold water in at bottom, hot out at top)
    T_cold_arr = np.asarray(T_cold, dtype=float)
    if T_cold_arr.ndim == 0:
        T_cold_arr = np.full(len(T_temp2), float(T_cold_arr))

    f = np.clip(params.alpha_draw * np.asarray(V_draw_l, dtype=float) / NODE_VOL_L,
                0.0, 1.0)  # (N,)

    T_new = T_temp2.copy()
    for i in range(3, 0, -1):
        T_new[:, i] = (1 - f) * T_temp2[:, i] + f * T_temp2[:, i - 1]
    T_new[:, 0] = (1 - f) * T_temp2[:, 0] + f * T_cold_arr

    T_new = np.clip(T_new, 5.0, 95.0)
    T_new = _soft_enforce_stratification_batch(T_new)
    return T_new


def tank_step(
    T: np.ndarray,
    Q_st_kwh: float,
    Q_ashp_kwh: float,
    Q_imm_kwh: float,
    T_amb: float,
    V_draw_l: float,
    T_cold: float,
    params: TankParams,
    dt_s: float = 1800.0,
    Q_dhw_kwh: float = 0.0,
) -> np.ndarray:
    """Advance the 4-node tank by one time step (Euler forward).

    Parameters
    ----------
    T : array of shape (4,) — current temperatures [°C].
    Q_st_kwh, Q_ashp_kwh, Q_imm_kwh : heat inputs this interval [kWh].
    T_amb : ambient temperature [°C].
    V_draw_l : draw-off volume this interval [L].
    T_cold : cold mains temperature [°C].
    params : TankParams instance.
    dt_s : time-step in seconds (default 1800 = 30 min).

    Returns
    -------
    T_new : updated temperatures (4,) [°C].
    """
    T = np.array(T, dtype=float)

    # Convert kWh → kJ for the interval
    Q_st_kj  = Q_st_kwh * 3600.0
    Q_ashp_kj = Q_ashp_kwh * 3600.0
    Q_imm_kj  = Q_imm_kwh * 3600.0
    Q_dhw_kj = Q_dhw_kwh * 3600.0

    # Limit effective input when upper layers are already hot.
    T_hot = float(np.max(T[1:]))
    Q_st_kj *= _st_acceptance_factor(T_hot)
    Q_ashp_kj *= _ashp_acceptance_factor(T_hot)

    # Normalise per-source split vectors so source energy is conserved.
    f_st = _normalise_fractions(params.f_st)
    f_ashp = _normalise_fractions(params.f_ashp)
    f_imm = _normalise_fractions(params.f_imm)

    # 1. Apply heat inputs
    T_temp = T.copy()
    for i in range(4):
        dQ = (
            f_st[i] * Q_st_kj
            + f_ashp[i] * Q_ashp_kj
            + f_imm[i] * Q_imm_kj
        )
        dT_heat = dQ / NODE_CAP
        T_temp[i] += dT_heat

    # 2. Apply standing losses, conduction, and mixing
    T_temp2 = T_temp.copy()
    for i in range(4):
        # Loss to ambient [kJ] = UA [kW/K] × ΔT [K] × dt [s]
        loss = params.UA_loss[i] * (T_temp[i] - T_amb) * dt_s

        # Adjacent-node conduction [kJ]
        cond = 0.0
        if i > 0:
            cond += params.UA_adj[i - 1] * (T_temp[i - 1] - T_temp[i]) * dt_s
        if i < 3:
            cond += params.UA_adj[i] * (T_temp[i + 1] - T_temp[i]) * dt_s

        # Draw-induced mixing (tendency toward neighbour average)
        mix = 0.0
        if i > 0:
            mix += params.mix_coeff * (T_temp[i - 1] - T_temp[i]) * dt_s
        if i < 3:
            mix += params.mix_coeff * (T_temp[i + 1] - T_temp[i]) * dt_s

        dT_loss = (-loss + cond + mix) / NODE_CAP
        T_temp2[i] = T_temp[i] + dT_loss

    for i in range(4):
        T_temp2[i] -= (Q_dhw_kj * DHW_EXTRACTION_FRACTIONS[i]) / NODE_CAP

    # 3. Apply draw-off displacement
    f = params.alpha_draw * V_draw_l / NODE_VOL_L
    f = np.clip(f, 0.0, 1.0)
    T_new = T_temp2.copy()
    # Displacement from top to bottom: hot water out top, cold in bottom
    for i in range(3, 0, -1):
        T_new[i] = (1 - f) * T_temp2[i] + f * T_temp2[i - 1]
    T_new[0] = (1 - f) * T_temp2[0] + f * T_cold

    # Enforce plausible bounds and remove any node inversions.
    T_new = np.clip(T_new, 5.0, 95.0)
    T_new = _soft_enforce_stratification_row(T_new)
    return T_new


def tank_step_with_breakdown(
    T: np.ndarray,
    Q_st_kwh: float,
    Q_ashp_kwh: float,
    Q_imm_kwh: float,
    T_amb: float,
    V_draw_l: float,
    T_cold: float,
    params: TankParams,
    dt_s: float = 1800.0,
    Q_dhw_kwh: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Advance one step and return detailed per-node energy breakdown.

    Energy terms are reported in kJ per node for this interval:
      heat_kj, loss_kj, cond_kj, mix_kj, draw_kj, net_kj.
    """
    T = np.array(T, dtype=float)

    Q_st_kj = Q_st_kwh * 3600.0
    Q_ashp_kj = Q_ashp_kwh * 3600.0
    Q_imm_kj = Q_imm_kwh * 3600.0
    Q_dhw_kj = Q_dhw_kwh * 3600.0

    # Apply same acceptance limiter as tank_step for consistency.
    T_hot = float(np.max(T[1:]))
    Q_st_kj *= _st_acceptance_factor(T_hot)
    Q_ashp_kj *= _ashp_acceptance_factor(T_hot)

    heat_kj = np.zeros(4)
    loss_kj = np.zeros(4)
    cond_kj = np.zeros(4)
    mix_kj = np.zeros(4)
    dhw_kj = np.zeros(4)

    f_st = _normalise_fractions(params.f_st)
    f_ashp = _normalise_fractions(params.f_ashp)
    f_imm = _normalise_fractions(params.f_imm)

    # 1) Heating contribution
    T_temp = T.copy()
    for i in range(4):
        dQ = (
            f_st[i] * Q_st_kj
            + f_ashp[i] * Q_ashp_kj
            + f_imm[i] * Q_imm_kj
        )
        heat_kj[i] = dQ
        T_temp[i] += dQ / NODE_CAP

    # 2) Loss, conduction, and mixing contribution
    T_temp2 = T_temp.copy()
    for i in range(4):
        loss = params.UA_loss[i] * (T_temp[i] - T_amb) * dt_s

        cond = 0.0
        if i > 0:
            cond += params.UA_adj[i - 1] * (T_temp[i - 1] - T_temp[i]) * dt_s
        if i < 3:
            cond += params.UA_adj[i] * (T_temp[i + 1] - T_temp[i]) * dt_s

        mix = 0.0
        if i > 0:
            mix += params.mix_coeff * (T_temp[i - 1] - T_temp[i]) * dt_s
        if i < 3:
            mix += params.mix_coeff * (T_temp[i + 1] - T_temp[i]) * dt_s

        loss_kj[i] = loss
        cond_kj[i] = cond
        mix_kj[i] = mix
        T_temp2[i] = T_temp[i] + (-loss + cond + mix) / NODE_CAP

    dhw_kj = Q_dhw_kj * DHW_EXTRACTION_FRACTIONS
    T_temp2 = T_temp2 - dhw_kj / NODE_CAP

    # 3) Draw displacement contribution (derived from node energy change)
    f = params.alpha_draw * V_draw_l / NODE_VOL_L
    f = np.clip(f, 0.0, 1.0)
    T_new = T_temp2.copy()
    for i in range(3, 0, -1):
        T_new[i] = (1 - f) * T_temp2[i] + f * T_temp2[i - 1]
    T_new[0] = (1 - f) * T_temp2[0] + f * T_cold

    draw_kj = (T_new - T_temp2) * NODE_CAP
    T_new = np.clip(T_new, 5.0, 95.0)
    T_new = _soft_enforce_stratification_row(T_new)

    total_kj = (T_new - T) * NODE_CAP
    net_kj = heat_kj - loss_kj + cond_kj + mix_kj - dhw_kj + draw_kj

    breakdown = {
        "T_prev": T.tolist(),
        "T_after_heat": T_temp.tolist(),
        "T_after_internal": T_temp2.tolist(),
        "T_new": T_new.tolist(),
        "heat_kj": heat_kj.tolist(),
        "loss_kj": loss_kj.tolist(),
        "cond_kj": cond_kj.tolist(),
        "mix_kj": mix_kj.tolist(),
        "dhw_kj": dhw_kj.tolist(),
        "draw_kj": draw_kj.tolist(),
        "net_kj": net_kj.tolist(),
        "total_kj_from_deltaT": total_kj.tolist(),
        "dT_nodes_c": (T_new - T).tolist(),
        "f_draw": float(f),
        "V_draw_l": float(V_draw_l),
        "Q_st_kwh": float(Q_st_kwh),
        "Q_ashp_kwh": float(Q_ashp_kwh),
        "Q_imm_kwh": float(Q_imm_kwh),
        "Q_dhw_kwh": float(Q_dhw_kwh),
    }
    return T_new, breakdown


def simulate(
    T0: np.ndarray,
    Q_st: np.ndarray,
    Q_ashp: np.ndarray,
    Q_imm: np.ndarray,
    T_amb: np.ndarray,
    V_draw: np.ndarray,
    T_cold: np.ndarray,
    params: TankParams,
    dt_s: float = 1800.0,
    Q_dhw: np.ndarray | None = None,
) -> np.ndarray:
    """Run the tank model over N time steps.

    Parameters
    ----------
    T0 : initial temperatures (4,).
    Q_st, Q_ashp, Q_imm : heat input arrays of shape (N,) [kWh per step].
    T_amb : ambient temperature array of shape (N,) [°C].
    V_draw : draw-off volume array of shape (N,) [L per step].
    T_cold : cold mains temperature array of shape (N,) [°C].
    params : TankParams.
    dt_s : time-step seconds.

    Returns
    -------
    T_hist : array (N+1, 4) — temperatures at each step (including T0).
    """
    N = len(Q_st)
    if Q_dhw is None:
        Q_dhw = np.zeros(N, dtype=float)
    T_hist = np.zeros((N + 1, 4))
    T_hist[0] = T0

    # Each step feeds the output of the previous step(T_hist[k]) as the input to the next (T_hist[k+1]).
    for k in range(N):
        T_hist[k + 1] = tank_step(
            T_hist[k],
            float(Q_st[k]),
            float(Q_ashp[k]),
            float(Q_imm[k]),
            float(T_amb[k]),
            float(V_draw[k]),
            float(T_cold[k]),
            params,
            dt_s,
            Q_dhw_kwh=float(Q_dhw[k]),
        )
    return T_hist
