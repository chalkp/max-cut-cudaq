# max-cut-qiskit

Qiskit fork of [max-cut-cudaq](https://github.com/chalkp/max-cut-cudaq): the same
graphs and LR-QAOA circuits, with the CUDA-Q kernels swapped for Qiskit Aer so
they can go to IBM hardware. The simulator-only studies (the full p-sweep,
fits, figures) stay upstream.

```
src/maxcut_qiskit/
  graphs.py     graph families, same as upstream
  qaoa.py       Problem: exact P(x*) and mean cut ratio, numpy or Aer
  aer.py        Aer circuit and simulator (cuStateVec), on-device readout
  cpu.py        cut values of all 2^n bitstrings, numpy state vector for small n
  schedules.py  gap-adapted and cubic schedules, COBYLA baseline angles
  ibm.py        device circuits: swap network or SABRE, ISA transpile, SamplerV2
experiments/
  hardware.py       sweep / pilot / depth-matched on an IBM device
  compare_cudaq.py  random circuits, states and sweep rows vs upstream
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
| `hardware` | `hardware.py` | sweep / pilot / depth-matched on an IBM device -> `hardware/` |
| `compare` | `compare_cudaq.py` | random circuits, states and sweep rows vs upstream -> `compare_cudaq.json` |

`pixi run <task>`, or `python experiments/<script>.py`.

`compare` runs the CUDA-Q side in upstream's environment: the superproject when
this repo is its `max-cut-qiskit` submodule, `../max-cut-cudaq` otherwise
(`--cudaq PATH` to override).

## Hardware

Save an IBM Quantum account once, in your own terminal:

```
pixi run python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; \
  S.save_account(channel='ibm_quantum_platform', token='<API key>', instance='<CRN>', set_as_default=True)"
```

```
pixi run hardware sweep --backend fake_fez --n 8 10 --p 1 2 3 5   # local noise model
pixi run hardware sweep --backend fake_fez --noiseless             # same circuits, ideal
pixi run hardware sweep --backend ibm_fez                          # estimate only
pixi run hardware sweep --backend ibm_fez --submit                 # job ids -> results/hardware
pixi run hardware collect results/hardware/sweep_ibm_fez.json
```

Dense graphs go through a swap network on a line of good qubits, sparse ones
through SABRE, whichever needs fewer two-qubit gates; fractional `rzz` is used
where the device has it. Dynamical decoupling on, twirling off (`--twirl`).

## Knobs

| variable | |
|---|---|
| `MAXCUT_BACKEND` | `auto` (numpy below ~14 dense qubits, else Aer), `aer`, `numpy` |
| `MAXCUT_PRECISION` | `fp64` (default) or `fp32`, Aer circuits only |
