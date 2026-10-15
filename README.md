# max-cut-cudaq

LR-QAOA (linear-ramp QAOA) for MaxCut. Exact results on graphs that reduce to
something small (ring, complete bipartite, complete), checked against CUDA-Q state
vector runs on the rest.

```
src/maxcut_cudaq/
  graphs.py    graph families + weighted complete graphs
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
| `pilot` | `gap_pilot.py` | gap location, gap-adapted vs linear schedule -> `gap_pilot.csv` |
| `depth-matched` | `depth_matched.py` | LR vs COBYLA at equal depth -> `depth_matched.csv` |

`pixi run <task>`, or `python experiments/<script>.py`.

## Knobs

| variable | |
|---|---|
| `MAXCUT_BACKEND` | `auto` (numpy below ~13 dense qubits, else CUDA-Q), `cudaq`, `numpy` |
| `MAXCUT_PRECISION` | `fp64` (default) or `fp32`, above 17 qubits |
| `MAXCUT_FUSION_N` | size the CUDA-Q fusion settings are tuned for at import (default 18) |
| `CUDAQ_FUSION_*` | override `maxcut_cudaq.fusion_env` |
