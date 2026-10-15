"""numpy side: cut values of every bitstring, and a fp64 state vector for small n.

Bit q of a basis index is qubit q, as in Qiskit.
"""

import numpy as np


def cut_values(n, src, tgt, w):
    """Cut weight of every bitstring in [0, 2^n), one pass per edge."""
    x = np.arange(1 << n)
    out = np.zeros(1 << n)
    for i, j, wij in zip(
        np.asarray(src).tolist(), np.asarray(tgt).tolist(), np.asarray(w, dtype=float).tolist()
    ):
        out += wij * (((x >> i) ^ (x >> j)) & 1)
    return out


def probabilities(n, hc, gammas, betas):
    """|psi|^2 of prod_k e^{+i beta_k sum X} e^{-i gamma_k H_C} |+>^n, H_C per bitstring."""
    psi = np.full(2**n, 2.0 ** (-n / 2), dtype=complex)
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * hc)
        t = psi.reshape((2,) * n)
        for ax in range(n):
            t = np.cos(b) * t + 1j * np.sin(b) * np.flip(t, axis=ax)
        psi = t.reshape(-1)
    return np.abs(psi) ** 2
