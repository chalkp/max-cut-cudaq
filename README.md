# max-cut-cudaq

LR-QAOA (linear-ramp QAOA) for MaxCut. Exact results on graphs that reduce to
something small (ring, complete bipartite, complete), checked against CUDA-Q state
vector runs on the rest.

```
src/maxcut_cudaq/
  graphs.py    graph families + weighted complete graphs
  exact.py     ring free-fermion formula, one- and two-spin models, gap-power schedules
  qaoa.py      Problem: circuit + exact P(x*) / mean cut ratio
  engine.py    CUDA-Q kernel, target selection, on-device readout
  gpu.py       cut values of all 2^n bitstrings, zero-copy state view
  cpu.py       numpy state vector for small n
experiments/   one script per study, writes to results/
```

## Install

Linux x86-64, NVIDIA GPU, CUDA 13 driver.

```
pixi install            # or: uv sync
```

## Run

| task | script | output |
|---|---|---|
| `sweep` | `lr_sweep.py` | LR p-sweep, n = 8..28, 7 families -> `lr_p_sweep.csv` |
| `fit` | `lr_sweep.py --fit` | eta(p) fits -> `lr_p_sweep_eta.csv` |
| `pilot` | `gap_pilot.py` | gap location, gap-adapted vs linear schedule -> `gap_pilot.csv` |
| `depth-matched` | `depth_matched.py` | LR vs COBYLA at equal depth -> `depth_matched.csv` |
| `q1-q5-q6` | `q1_q5_q6.py` | Q1 schedule depths, Q5 best Delta, Q6 threshold -> `q5_q1_numbers.json` |
| `limits` | `delta_limits.py` | the three limits on Delta_gamma -> `q5_q1_numbers.json` |
| `solvable` | `solvable_models.py` | ring / spin model checks -> `verification_numbers.json` |
| `figures` | `figures.py` | `results/figures/` |

`pixi run <task>`, or `python experiments/<script>.py`. `solvable` and `figures`
read the sweep and pilot output, so run those first.

The full sweep is ~5 GPU-hours on an RTX 3080 (`pixi run sweep --hours 6`). It
restarts its worker every few hundred circuits and resumes from the CSV, so it can be
killed and restarted. The hour budget is stored in `results/lr_p_sweep.csv.deadline`;
delete it to start a new one.

## Knobs

| variable | |
|---|---|
| `MAXCUT_BACKEND` | `auto` (numpy below ~13 dense qubits, else CUDA-Q), `cudaq`, `numpy` |
| `MAXCUT_PRECISION` | `fp64` (default) or `fp32`, above 17 qubits |
| `MAXCUT_FUSION_N` | size the CUDA-Q fusion settings are tuned for at import (default 18) |
| `CUDAQ_FUSION_*` | override `maxcut_cudaq.fusion_env` |
