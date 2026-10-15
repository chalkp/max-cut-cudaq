"""numpy side: cut values of every bitstring, and a fp64 state vector for small n.

Bit q of a basis index is qubit q, as in Qiskit.
"""

import numpy as np


def _spins(k):
    # z = 1 - 2b for the k low bits of 0 .. 2^k - 1, shape (2^k, k)
    x = np.arange(1 << k)
    return (1 - 2 * ((x[:, None] >> np.arange(k)) & 1)).astype(float)


def cut_values(n, src, tgt, w):
    """cut_values() - cut weight of every bitstring in [0, 2^n)
    @n: number of qubits
    @src: edge tails
    @tgt: edge heads
    @w: edge weights

    cut = (sum w - sum w z_i z_j) / 2. Split the qubits into a low half L and
    a high half H: edges inside a half give one vector per half, and edges
    across give z_H C z_L^T, one matrix product for the whole (2^|H|, 2^|L|)
    table instead of |E| passes over 2^n. Integer weights stay exact.
    """
    src, tgt = np.asarray(src, dtype=np.int64), np.asarray(tgt, dtype=np.int64)
    w = np.asarray(w, dtype=float)
    k = n // 2
    zl, zh = _spins(k), _spins(n - k)
    lo, hi = (src < k) & (tgt < k), (src >= k) & (tgt >= k)
    cross = ~(lo | hi)
    a, b = np.minimum(src, tgt)[cross], np.maximum(src, tgt)[cross]
    C = np.zeros((n - k, k))
    np.add.at(C, (b - k, a), w[cross])
    out = zh @ (C @ zl.T)
    out += ((zh[:, src[hi] - k] * zh[:, tgt[hi] - k]) @ w[hi])[:, None]
    out += (zl[:, src[lo]] * zl[:, tgt[lo]]) @ w[lo]
    out *= -0.5
    out += 0.5 * w.sum()
    return out.reshape(-1)


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
