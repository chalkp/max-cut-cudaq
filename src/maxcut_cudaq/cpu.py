"""Plain numpy state vector, fp64, one core. Used for small circuits where the GPU loses."""

import numpy as np


def hc_vector(n, src, tgt, wn):
    """H_C = sum wn Z_i Z_j for every bitstring."""
    x = np.arange(2**n)
    z = 1 - 2 * ((x[:, None] >> np.arange(n)) & 1)
    return (z[:, src] * z[:, tgt]) @ wn


def probabilities(n, hc, gammas, betas):
    """|psi|^2 of prod_k e^{+i beta_k sum X} e^{-i gamma_k H_C} |+>^n."""
    psi = np.full(2**n, 2.0 ** (-n / 2), dtype=complex)
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * hc)
        t = psi.reshape((2,) * n)
        for ax in range(n):
            t = np.cos(b) * t + 1j * np.sin(b) * np.flip(t, axis=ax)
        psi = t.reshape(-1)
    return np.abs(psi) ** 2
