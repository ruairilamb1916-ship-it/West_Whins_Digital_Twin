# DHW Digital Twin Stage 1

This repository fits a Stage 1 digital twin for a domestic hot water system with a stratified tank and an ASHP performance model. The main goal is to identify tank behaviour and ASHP maps from historical half-hourly data, then inspect how the fitted COP map compares with label-based COP targets.
# West Whins Digital Twin – ASHP COP Calibration

## Overview
This project calibrates an Air Source Heat Pump (ASHP) model using measured data.

Recent work focused on improving COP (Coefficient of Performance) estimation:
- Separate capacity and COP labels
- Clean filtering of COP training data
- Weighted COP regression to better match real operation

---

## Key Results

| Season | COP_fit | COP_target |
|-------|--------|-----------|
| Winter (w0) | ~1.80 | ~1.92 |
| Spring (w12) | ~2.19 | ~2.31 |
| Summer (w24) | ~2.35 | ~2.23 |

→ Model now tracks seasonal COP behaviour much better.

---

## Improvements Made

- Introduced `q_mix` (blended heat label)
- Separated COP vs capacity training signals
- Added COP filtering logic
- Added soft weighting (power = 1.3)
- Stabilised regression with clipping + regularisation

---

## Diagnostics

### COP Fit vs Target
![COP comparison](output/plots/cop_comparison.png)

---

## How to Run

```bash
python3 run_stage1.py --fit-profile fast_ashp
python3 scripts/debug_ashp_map.py

## Model structure

- Tank model: 4-node DHW tank with standing loss, mixing, draws, immersion, and solar thermal input.
- ASHP maps: separate fitted maps for electrical power, delivered heat capacity, and direct COP as functions of outdoor temperature and sink temperature.

## Run stage 1

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main identification pipeline:

```bash
python3 run_stage1.py --fit-profile fast_ashp --start-week 0 --weeks 12
```

Useful seasonal comparisons:

```bash
python3 run_stage1.py --fit-profile fast_ashp --start-week 12 --weeks 12
python3 run_stage1.py --fit-profile fast_ashp --start-week 24 --weeks 12
```

## Run ASHP map debugging

After stage 1 has produced `output/params.json` and `output/debug_labels_fast.csv`, run:

```bash
python3 scripts/debug_ashp_map.py --start-week 0 --weeks 12
```

This writes `output/debug_ashp_map.csv` for the same seasonal subset used in `run_stage1.py`.

## soft_weighted_cop_v1 baseline

The current baseline uses:

- `Q_back` as the broad capacity-fit target
- `q_mix` / `q_mix_cop` as the COP-fit target
- clean COP filtering applied only to the COP target series
- clipped COP targets before regression
- soft COP sample weighting with a 1.3 power to give warmer, higher-COP points more influence
- light L2 regularisation to stabilise the COP coefficients

This setup is intended to improve warm-period COP fit without making the quadratic COP map unstable.

## Key outputs

- `output/params.json`: fitted tank and ASHP parameters
- `output/summary.json` or `output/summary_fast_ashp.json`: summary metrics including validation RMSE
- `output/debug_labels_fast.csv`: blended ASHP heat labels used for debugging
- `output/debug_ashp_map.csv`: per-timestep comparison of `COP_fit`, `COP_target`, and fitted heat/power

The main diagnostics to compare are:

- `COP_fit` vs `COP_target`
- validation RMSE reported by `run_stage1.py`
- mean `COP_fit` and mean `COP_target` reported by `debug_ashp_map.py`
