# max-cut-qiskit

Qiskit fork of [max-cut-cudaq](https://github.com/chalkp/max-cut-cudaq) for
running LR-QAOA on UTokyo's IBM Quantum System One (`ibm_kawasaki`). Device
counts are scored against exact Qiskit Aer runs of the same angles, and both are
cross-checked against upstream's CUDA-Q results. The simulator-only studies
(the full p-sweep, fits, figures) stay upstream.

```
src/maxcut_qiskit/
  graphs.py     graph families, same as upstream
  qaoa.py       Problem: exact P(x*) and mean cut ratio, numpy or Aer
  aer.py        Aer circuit and simulator (cuStateVec), on-device readout
  cpu.py        cut values of all 2^n bitstrings, numpy state vector for small n
  schedules.py  gap-adapted and cubic schedules, COBYLA baseline angles
  ibm.py        device circuits, routing, readout correction, local noise models
experiments/
  hardware.py       sweep / pilot / depth-matched on ibm_kawasaki or its model
  compare_cudaq.py  random circuits, states and sweep rows vs upstream
```

## Install

Linux x86-64. An NVIDIA GPU is used when there is one (`qiskit-aer-gpu`,
CUDA 12 runtime); without one Aer runs on the CPU. Both lock files resolve
against packages published before 2026-10-26 (`exclude-newer`).

```
pixi install            # or: uv sync
```

`compare` runs the CUDA-Q side in upstream's environment: the superproject when
this repo is its `max-cut-qiskit` submodule, `../max-cut-cudaq` otherwise
(`--cudaq PATH` to override).

## The device

`ibm_kawasaki` is the System One in Kawasaki that the University of Tokyo runs
for the QII consortium: a 127-qubit Eagle r3 (ECR) since 2023, announced for a
156-qubit Heron (CZ, fractional `rzz`) in 2025. Gates, couplings and errors are
read off the device, so either works. `fake_kawasaki` is Qiskit's Eagle r3
snapshot. Save the QII account once, in your own terminal:

```
pixi run python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; \
  S.save_account(channel='ibm_quantum_platform', token='<API key>', instance='<QII instance>', set_as_default=True)"
pixi run python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; print([b.name for b in S().backends()])"
```

## Run

```
pixi run hardware sweep                                     # fake_kawasaki, local
pixi run hardware sweep --noiseless                         # same circuits, ideal
pixi run compare --hardware results/hardware/sweep_fake_kawasaki_ideal.json
pixi run hardware sweep --backend ibm_kawasaki --simulate   # live calibration, local
pixi run hardware sweep --backend ibm_kawasaki              # estimate only
pixi run hardware sweep --backend ibm_kawasaki --submit     # job ids -> results/hardware
pixi run hardware collect results/hardware/sweep_ibm_kawasaki.json
pixi run compare                                            # circuits vs upstream
```

- Each circuit is built twice, as a swap network on the best line of qubits and
  through SABRE; the one more likely to run error-free (`fidelity`, product of
  1 - error over its gates and readouts) is kept.
- Circuits under `--min-fid` (0.01) are dropped. The grids ask the questions and
  the device decides which it can answer, so a Heron keeps more than an Eagle.
- Dynamical decoupling (XpXm) on, twirling off (`--twirl`), one Batch
  (`--mode job` where batches are not allowed).
- Two readout calibration circuits ride along; `*_mitigated` columns are the
  counts after the tensored inverse.
- Local noisy runs use a density matrix up to 14 qubits and shot trajectories
  above (2-4 min a circuit on 16-20 qubits). SABRE may route through spare
  qubits, so a circuit can touch more qubits than it reads.

On `fake_kawasaki` (ECR median error 9.0e-3, readout 1.6%), default grids,
4000 shots:

| study | circuits kept | QPU estimate | noiseless vs CUDA-Q, z sd | noisy model |
|---|---|---|---|---|
| sweep | 156 of 350 | 3.5 min | 1.00 | above random (3 sigma) in 140; rings peak at p = 4-10, exact still rising at the deepest p kept |
| pilot | 180 of 400 | 4.3 min | 1.01 | gap-adapted beats linear in 36 of 45 groups (exact: 40) |
| depth-matched | 72 of 168 | 1.7 min | 1.15 | 36 pairs left, mostly p = 3, 5; LR wins 17 (exact: 14) |

QPU estimates leave out job overheads. A noisy circuit keeps about F^0.7 of
P(x*); the readout correction lifts that from 0.32 to 0.34 on average.

## Knobs

| variable | |
|---|---|
| `MAXCUT_BACKEND` | `auto` (numpy below ~14 dense qubits, else Aer), `aer`, `numpy` |
| `MAXCUT_PRECISION` | `fp64` (default) or `fp32`, Aer circuits only |
