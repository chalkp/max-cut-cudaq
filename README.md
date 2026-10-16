# max-cut-qiskit

Qiskit fork of [max-cut-cudaq](https://github.com/chalkp/max-cut-cudaq): the same
graphs and LR-QAOA circuits, with the CUDA-Q kernels swapped for Qiskit Aer so
they can go to IBM hardware. The simulator-only studies (the full p-sweep,
fits, figures) stay upstream.

```
src/maxcut_qiskit/
  graphs.py    graph families, same as upstream
  qaoa.py      Problem: exact P(x*) and mean cut ratio, numpy or Aer
  aer.py       Aer circuit and simulator (cuStateVec), on-device readout
  cpu.py       cut values of all 2^n bitstrings, numpy state vector for small n
experiments/   one script per study, writes to results/
```

## Install

Linux x86-64. An NVIDIA GPU is used when there is one (`qiskit-aer-gpu-cu11`,
CUDA 11 runtime); without one Aer runs on the CPU. Both lock files resolve
against packages published before 2026-10-15 (`exclude-newer`).

```
pixi install            # or: uv sync
```

## Run

| task | script | output |
|---|---|---|
| `pilot` | `gap_pilot.py` | gap location, gap-adapted vs linear schedule -> `gap_pilot.csv` |
| `depth-matched` | `depth_matched.py` | LR vs COBYLA at equal depth -> `depth_matched.csv` |
| `compare` | `compare_cudaq.py` | random circuits, states and sweep rows vs upstream -> `compare_cudaq.json` |

`pixi run <task>`, or `python experiments/<script>.py`.

`compare` runs the CUDA-Q side in upstream's environment: the superproject when
this repo is its `max-cut-qiskit` submodule, `../max-cut-cudaq` otherwise
(`--cudaq PATH` to override).

## Knobs

| variable | |
|---|---|
| `MAXCUT_BACKEND` | `auto` (numpy below ~14 dense qubits, else Aer), `aer`, `numpy` |
| `MAXCUT_PRECISION` | `fp64` (default) or `fp32`, Aer circuits only |
