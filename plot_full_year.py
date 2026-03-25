#!/usr/bin/env python3
"""Plot full year tank temps: measured vs teacher-forced vs closed-loop.

Loads already-fitted parameters from output/params.json so identification
does not need to be re-run.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src import data_loader, identification, ashp_model, tank_model

ROOT = Path(__file__).resolve().parent
csv_path  = ROOT / 'data' / 'FullDS_Findhorn.csv'
yaml_path = ROOT / 'column_mapping.yaml'
params_path = ROOT / 'output' / 'params.json'

# ── Load fitted parameters ────────────────────────────────────────────────────
print("Loading fitted parameters from params.json ...")
with open(params_path) as fh:
    saved = json.load(fh)

tank_p = tank_model.TankParams()
tank_p.UA_loss    = np.array(saved['tank']['UA_loss'])
tank_p.UA_adj     = np.array(saved['tank']['UA_adj'])
tank_p.f_st       = np.array(saved['tank']['f_st'])
tank_p.f_ashp     = np.array(saved['tank']['f_ashp'])
tank_p.f_imm      = np.array(saved['tank']['f_imm'])
tank_p.mix_coeff  = float(saved['tank']['mix_coeff'])
tank_p.alpha_draw = float(saved['tank']['alpha_draw'])
tank_p.T_mains    = float(saved['tank']['T_mains'])

ashp_p = ashp_model.ASHPParams()
ashp_p.a = np.array(saved['ashp']['a'])
ashp_p.b = np.array(saved['ashp']['b'])

# ── Load & split data (same 70/30 split used during fitting) ─────────────────
print("Loading data...")
df = data_loader.load_and_clean(csv_path, yaml_path)
df = df.dropna(subset=['tank_bottom_c', 'tank_mid_c', 'tank_mid_hi_c', 'tank_top_c'], how='all')
df['st_kwh'] = __import__('src.solar_thermal', fromlist=['compute_st_energy']).compute_st_energy(df)
print(f'Full data: {len(df)} rows, {df.index.min()} to {df.index.max()}')

split_idx = int(len(df) * 0.7)
df_val = df.iloc[split_idx:].copy()
print(f'Val set: {len(df_val)} rows from {df_val.index[0]} to {df_val.index[-1]}')

# ── Prepare inputs ────────────────────────────────────────────────────────────
print("Preparing inputs...")
inputs = identification.prepare_inputs(df_val, ashp_p)
T_val = inputs['T_meas']
print(f'T_val shape: {T_val.shape}, min={T_val.min():.1f}, max={T_val.max():.1f}')

# ── Teacher-forced simulation ─────────────────────────────────────────────────
print("Running teacher-forced simulation...")
T_tf = np.zeros_like(T_val)
T_tf[0] = T_val[0]
for k in range(len(T_val) - 1):
    T_tf[k + 1] = tank_model.tank_step(
        T_val[k],
        float(inputs['Q_st'][k]),
        float(inputs['Q_ashp'][k]),
        float(inputs['Q_imm'][k]),
        float(inputs['T_amb'][k]),
        float(inputs['V_draw'][k]),
        float(inputs['T_cold'][k]),
        tank_p,
    )
print(f'T_tf: min={T_tf.min():.1f}, max={T_tf.max():.1f}')

# ── Closed-loop simulation ────────────────────────────────────────────────────
print("Running closed-loop simulation...")
T_cl = identification.simulate_closed_loop(
    T_val[0],
    inputs['Q_st'],
    inputs['Q_imm'],
    inputs['T_amb'],
    inputs['V_draw'],
    inputs['T_cold'],
    inputs['T_out'],
    inputs['P_meas'],
    ashp_p,
    tank_p,
)
print(f'T_cl: min={T_cl.min():.1f}, max={T_cl.max():.1f}')

# ── RMSE summary ──────────────────────────────────────────────────────────────
def rmse(a, b):
    return np.sqrt(np.nanmean((a - b) ** 2))

rmse_tf = rmse(T_tf[1:], T_val[1:])
rmse_cl = rmse(T_cl[1:len(T_val) + 1], T_val)
print(f'RMSE  teacher-forced: {rmse_tf:.3f} °C')
print(f'RMSE  closed-loop:    {rmse_cl:.3f} °C')

# ── Plot ──────────────────────────────────────────────────────────────────────
print("Creating plot...")
labels = ['bottom', 'mid', 'mid-hi', 'top']
fig, axs = plt.subplots(4, 1, figsize=(18, 14), sharex=True)

for i, ax in enumerate(axs):
    ax.plot(df_val.index, T_val[:, i],
            label='measured', color='black', linewidth=0.6, alpha=0.8, zorder=3)
    ax.plot(df_val.index, T_tf[:, i],
            label=f'teacher-forced (RMSE {rmse(T_tf[1:, i], T_val[1:, i]):.2f} °C)',
            color='steelblue', linestyle='--', linewidth=0.9, alpha=0.85)
    ax.plot(df_val.index, T_cl[1:len(T_val) + 1, i],
            label=f'closed-loop (RMSE {rmse(T_cl[1:len(T_val)+1, i], T_val[:, i]):.2f} °C)',
            color='tomato', linestyle=':', linewidth=0.9, alpha=0.85)
    ax.set_title(f'Node: {labels[i]}', fontsize=11)
    ax.set_ylabel('Temp [°C]', fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', fontsize=8)

axs[-1].set_xlabel('Date', fontsize=10)
fig.suptitle(
    f'Tank Temperature – Validation period\n'
    f'Teacher-forced RMSE: {rmse_tf:.2f} °C   |   Closed-loop RMSE: {rmse_cl:.2f} °C',
    fontsize=13,
)
fig.tight_layout()

out = ROOT / 'output' / 'plots' / 'tank_temp_full_year.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches='tight')
print(f'Saved to {out}')
plt.close()
print("Done!")
