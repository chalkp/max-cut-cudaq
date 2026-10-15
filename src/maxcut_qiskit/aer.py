"""Aer side of qaoa.py: the circuit, the simulator and on-device readout.

Imported only once a circuit really runs on Aer, so numpy-only work never
touches the GPU.
"""

from functools import lru_cache

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator


def circuit(P, gammas, betas):
    """H on every qubit, then per layer rzz(2 gamma wn) per edge and rx(-2 beta)."""
    qc = QuantumCircuit(P.n)
    qc.h(range(P.n))
    edges = list(zip(P.src.tolist(), P.tgt.tolist(), P.wn.tolist()))
    for g, b in zip(gammas, betas):
        for i, j, x in edges:
            qc.rzz(2.0 * float(g) * x, i, j)
        qc.rx(-2.0 * float(b), range(P.n))
    return qc


@lru_cache
def simulator(precision):
    """State vector AerSimulator, on the GPU when there is one."""
    gpu = "GPU" in AerSimulator().available_devices()
    return AerSimulator(
        method="statevector",
        device="GPU" if gpu else "CPU",
        precision="single" if precision == "fp32" else "double",
    )


def device():
    return simulator("fp64").options.device.lower()


def run(qc, precision):
    res = simulator(precision).run(qc).result()
    if not res.success:
        raise RuntimeError(f"aer: {res.status}")
    return res.data()


def evaluate(P, gammas, betas, precision):
    """evaluate() - P(x*) and the expected cut, read off on the device
    @P: the Problem
    @gammas: cost angles, one per layer
    @betas: mixer angles, one per layer
    @precision: 'fp32' or 'fp64'

    Only the optimal amplitudes and <sum w ZZ> come back, never the 2^n state.
    """
    qc = circuit(P, gammas, betas)
    qc.save_amplitudes_squared(P.optimal_indices().tolist(), label="pstar")
    zz = SparsePauliOp.from_sparse_list(
        [("ZZ", [i, j], x) for i, j, x in zip(P.src.tolist(), P.tgt.tolist(), P.w.tolist())],
        num_qubits=P.n,
    )
    qc.save_expectation_value(zz, range(P.n), label="zz")
    data = run(qc, precision)
    return float(np.sum(data["pstar"])), 0.5 * (float(P.w.sum()) - float(np.real(data["zz"])))


def probabilities(P, gammas, betas, precision):
    qc = circuit(P, gammas, betas)
    qc.save_probabilities(label="pr")
    return np.asarray(run(qc, precision)["pr"], dtype=float)
