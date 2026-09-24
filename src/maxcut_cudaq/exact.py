"""Exact LR-QAOA on the graphs that reduce to something small. No 2^n here.

Ring C_n, n even: free fermions. Flip one sublattice, Jordan-Wigner, and the
circuit falls apart into n/2 two-level systems, k_m = (2m - 1) pi / n. A layer
acts on mode k as e^{+2i beta tz} e^{-2i gamma (cos k tz - sin k tx)}, starting
from the vacuum, and P(x*) = prod_m P(k_m). O(n p).

K(n/2, n/2): two spins of size n/4, H_C = 4 Sz_A Sz_B, dimension (n/2 + 1)^2.

K_n: one spin n/2, H_C = 2 Sz^2 - n/2, dimension n + 1.

Schedules are layer boundaries u[0..p]: gamma_k = u[k+1] dg, beta_k = (1 - u[k]) db.
The linear ramp is u_k = k / p.
"""

import numpy as np

DG, DB = 0.6, 0.3


def linear_u(p):
    return np.linspace(0.0, 1.0, p + 1)


def angles(u, dg=DG, db=DB):
    u = np.asarray(u, dtype=float)
    return u[1:] * dg, (1 - u[:-1]) * db


def lr_schedule(p, dg=DG, db=DB):
    i = np.arange(p)
    return (i + 1) / p * dg, (1 - i / p) * db


# ----------------------------------------------------------------------------- ring
def momenta(n):
    if n % 2:
        raise ValueError("even n only (odd rings are frustrated)")
    return (2 * np.arange(1, n // 2 + 1) - 1) * np.pi / n


def mode_probabilities(k, gammas, betas):
    """Probability that mode k ends in its ground state, for an array of k."""
    k = np.asarray(k, dtype=float)
    c, s = np.cos(k), np.sin(k)
    a = np.ones_like(k, dtype=complex)  # vacuum amplitude
    b = np.zeros_like(k, dtype=complex)  # pair amplitude
    for g, bt in zip(gammas, betas):
        cg, sg = np.cos(2 * g), np.sin(2 * g)
        a, b = (
            (cg - 1j * sg * c) * a + (1j * sg * s) * b,
            (1j * sg * s) * a + (cg + 1j * sg * c) * b,
        )
        a, b = np.exp(2j * bt) * a, np.exp(-2j * bt) * b
    return np.abs(np.sin(k / 2) * a + np.cos(k / 2) * b) ** 2


def ring_log2_success(n, p, dg=DG, db=DB, u=None):
    """log2 P(x*) on the n-ring. Use this one once P underflows."""
    g, b = lr_schedule(p, dg, db) if u is None else angles(u, dg, db)
    return float(np.sum(np.log2(mode_probabilities(momenta(n), g, b))))


def ring_success(n, p, dg=DG, db=DB, u=None):
    return 2.0 ** ring_log2_success(n, p, dg, db, u)


def eta_infinite(p, dg=DG, db=DB, N=1 << 14, u=None):
    """eta(p) as n -> inf, from the slope between N and 2N.

    Don't replace this with a midpoint rule on the k integral; the log
    singularity at k = 0 biases it.
    """
    return (ring_log2_success(N, p, dg, db, u) - ring_log2_success(2 * N, p, dg, db, u)) / N


def adiabatic_schedule(n, p, dg=DG, db=DB, grid=20001):
    """Gap-adapted ring schedule, du/dt ~ A^(-1/2) with A(u) = sum_k |B_k x B_k'|^2 / (4 |B_k|^4)."""
    k = momenta(n)[:, None]
    u = np.linspace(0, 1, grid)[None, :]
    bz, bx = -2 * db * (1 - u) + 2 * dg * u * np.cos(k), -2 * dg * u * np.sin(k)
    dbz, dbx = 2 * db + 2 * dg * np.cos(k), -2 * dg * np.sin(k)
    rate = np.sqrt(np.sum((bz * dbx - bx * dbz) ** 2 / (4 * (bz**2 + bx**2) ** 2), axis=0))
    tau = np.concatenate([[0], np.cumsum((rate[1:] + rate[:-1]) / 2)])
    return np.interp(np.linspace(0, 1, p + 1), tau / tau[-1], u[0])


def lowest_mode_gap(n, u, kind="continuum", dg=DG, db=DB):
    """Gap of the slowest mode k = pi/n, 'continuum' or per-layer quasi-energy ('floquet')."""
    k1 = np.pi / n
    if kind == "continuum":
        bz, bx = -2 * db * (1 - u) + 2 * dg * u * np.cos(k1), -2 * dg * u * np.sin(k1)
        return 2 * np.sqrt(bz**2 + bx**2)
    g, b = u * dg, (1 - u) * db
    c = np.cos(2 * g) * np.cos(2 * b) + np.sin(2 * g) * np.sin(2 * b) * np.cos(k1)
    return 2 * np.arccos(np.clip(c, -1, 1))


class GapPower:
    """GapPower - ring schedule with du/dt ~ gap(u)^q
    @n: ring size
    @q: gap exponent
    @kind: 'continuum' or 'floquet' gap
    @taper: False, True (3t^2 - 2t^3) or 'smoother' (6t^5 - 15t^4 + 10t^3);
            the taper slows the clock to zero at both ends

    Call it with p to get u[0..p].
    """

    TAPERS = {
        False: lambda t: t,
        True: lambda t: 3 * t**2 - 2 * t**3,
        "smoother": lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3,
    }

    def __init__(self, n, q, kind="continuum", taper=False, grid=400001, dg=DG, db=DB):
        self.u = np.linspace(0, 1, grid)
        rate = lowest_mode_gap(n, self.u, kind, dg, db) ** (-q)
        tau = np.concatenate([[0], np.cumsum((rate[1:] + rate[:-1]) / 2)])
        self.t = tau / tau[-1]
        self.taper = taper

    def __call__(self, p):
        return np.interp(self.TAPERS[self.taper](np.linspace(0, 1, p + 1)), self.t, self.u)

    def end_fraction(self):
        """Fraction of layers spent at u < 0.1 or u > 0.9."""
        return float(np.interp(0.1, self.u, self.t) + 1 - np.interp(0.9, self.u, self.t))


def power_ramp(p, r, ustar):
    """Ramp that crosses the critical point u* like |t - u*|^r."""
    t = np.linspace(0, 1, p + 1)
    lo = ustar - ustar * ((ustar - np.minimum(t, ustar)) / ustar) ** r
    hi = ustar + (1 - ustar) * ((np.maximum(t, ustar) - ustar) / (1 - ustar)) ** r
    return np.where(t < ustar, lo, hi)


def first_depth(n, make_u, target=-1.0, dg=DG, db=DB, start=2):
    """first_depth() - smallest depth that reaches a target on the ring
    @n: ring size
    @make_u: p -> u[0..p]
    @target: log2 P(x*) to reach, default -1 (P = 1/2)

    Walks p up in 5% steps, so the answer is good to 5%.
    """
    p = max(2, start)
    while ring_log2_success(n, p, dg, db, u=make_u(p)) < target:
        p = max(p + 1, int(p * 1.05))
    return p


# ----------------------------------------------------------------------------- large spins
def spin_ops(m):
    """Sz (diagonal) and Sx (matrix) of spin m/2, basis M = m/2 .. -m/2."""
    S = m / 2
    M = S - np.arange(m + 1)
    off = 0.5 * np.sqrt(S * (S + 1) - M[:-1] * (M[:-1] - 1))
    return M, np.diag(off, 1) + np.diag(off, -1)


def bipartite_state(n, p, dg=DG, db=DB, u=None):
    """Probabilities over (M_A, M_B) after the circuit on K(n/2, n/2)."""
    m = n // 2
    M, Sx = spin_ops(m)
    w, V = np.linalg.eigh(Sx)
    plus = V[:, np.argmax(w)]  # S^x = +m/2, i.e. |+>^m
    psi = np.outer(plus, plus).astype(complex)
    zz = 4 * np.outer(M, M)
    uu = linear_u(p) if u is None else np.asarray(u)
    for k in range(p):
        psi = psi * np.exp(-1j * uu[k + 1] * dg * zz)
        R = (V * np.exp(2j * (1 - uu[k]) * db * w)) @ V.T  # e^{+2i beta S^x}
        psi = R @ psi @ R.T
    return np.abs(psi) ** 2


def bipartite_success(n, p, dg=DG, db=DB, u=None):
    q = bipartite_state(n, p, dg, db, u)
    return float(q[0, -1] + q[-1, 0])


def complete_run(n, p, dg=DG, db=DB):
    """(P(x*), <Sz^2> / random) on K_n. Max cuts are the balanced ones, M = 0."""
    M, Sx = spin_ops(n)
    w, V = np.linalg.eigh(Sx)
    psi = V[:, np.argmax(w)].astype(complex)
    zz = 2 * M**2 - n / 2
    for k in range(p):
        psi = psi * np.exp(-1j * (k + 1) / p * dg * zz)
        psi = V @ (np.exp(2j * (1 - k / p) * db * w) * (V.T @ psi))
    pr = np.abs(psi) ** 2
    return float(pr[n // 2]), float(pr @ M**2) / (n / 4)


def complete_success(n, p, dg=DG, db=DB):
    return complete_run(n, p, dg, db)[0]
