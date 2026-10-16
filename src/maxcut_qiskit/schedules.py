"""Angle schedules other than the plain ramp.

The pilot schedules walk the same path H(u) = (1 - u) db H_M + u dg H_C, with
the same endpoints and depth, and differ only in how fast u moves:

  linear       u(t) = t
  adiabatic    du/dt ~ A(u)^(-1/2), A the leak weight out of the ground state.
               It needs the spectrum, so it is an oracle: the most that knowing
               the gap can buy.
  slow_middle  u(t) = (1 + (2t - 1)^3) / 2
  slow_ends    u(t) = 3t^2 - 2t^3

cobyla() is the depth-matched baseline, angles optimised on the exact state.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import minimize

from .qaoa import DB, DG

U_GRID = np.linspace(0.0, 1.0, 51)
NAMES = ("linear", "adiabatic", "slow_middle", "slow_ends")


def _even_sector(n, hc):
    # Both terms keep the global flip prod X and |+>^n is even under it, so
    # work on (|x> + |~x>) / sqrt2 with the top bit of x clear.
    N, half = 2**n, 2 ** (n - 1)
    x = np.arange(half)
    rows, cols = [], []
    for j in range(n):
        y = x ^ (1 << j)
        rows.append(x)
        cols.append(np.where(y >= half, (~y) & (N - 1), y))
    HM = -sp.coo_matrix(
        (np.ones(n * half), (np.concatenate(rows), np.concatenate(cols))), shape=(half, half)
    ).tocsr()
    return HM, hc[:half]


def leak_weight(P, k_extra=10):
    """leak_weight() - A(u) on U_GRID for one instance
    @P: the Problem
    @k_extra: eigenpairs computed above the optimal manifold

    The lowest m = (optimal bitstrings) / 2 even levels all end up optimal and
    leaking among them is free, so A = sum_{j>=m} |V_j0|^2 / (E_j - E_0)^2
    with V = dH/du.

    XXX: near u = 0 the first excited level is hugely degenerate and only
    m + k_extra pairs are computed, so eigsh's random start decides which part
    of the multiplet lands in A. The adiabatic arm is not reproducible there.
    """
    HM, hc = _even_sector(P.n, P.hc_host())
    m = int(P.optimal_mask_host().sum()) // 2
    k = min(m + k_extra, hc.size - 2)
    V = (-DB) * HM + DG * sp.diags(hc)
    A = np.full(U_GRID.size, np.nan)
    v0 = None
    for i, u in enumerate(U_GRID):
        H = (1 - u) * DB * HM + u * DG * sp.diags(hc)
        for attempt in range(3):
            try:
                E, W = spla.eigsh(H, k=k, which="SA", v0=v0, tol=1e-9, maxiter=20000)
                break
            except spla.ArpackNoConvergence:
                if attempt == 2:
                    raise
                v0 = None  # the warm start sometimes stalls it
        o = np.argsort(E)
        E, W = E[o], W[:, o]
        v0 = W[:, 0]
        c = np.abs(W[:, m:].T @ (V @ W[:, 0]))
        dE = E[m:] - E[0]
        ok = (c > 1e-7) & (dE > 1e-10)
        if ok.any():
            A[i] = np.sum(c[ok] ** 2 / dE[ok] ** 2)
    return A


def schedule(name, p, A=None):
    """u at the layer boundaries t = 0, 1/p, ..., 1; 'adiabatic' needs leak_weight()."""
    t = np.linspace(0, 1, p + 1)
    if name == "linear":
        return t
    if name == "slow_middle":
        return (1 + (2 * t - 1) ** 3) / 2
    if name == "slow_ends":
        return 3 * t**2 - 2 * t**3
    if name == "adiabatic":
        good = np.isfinite(A)
        rate = np.sqrt(np.interp(U_GRID, U_GRID[good], A[good]))  # dt/du, up to a constant
        fine = np.linspace(0, 1, 2001)
        r = np.interp(fine, U_GRID, rate)
        tau = np.concatenate([[0], np.cumsum((r[1:] + r[:-1]) / 2 * np.diff(fine))])
        return np.interp(t, tau / tau[-1], fine)
    raise ValueError(name)


def cobyla(P, p, seed):
    """cobyla() - QAOA angles optimised on the exact state
    @P: the Problem
    @p: number of layers
    @seed: seeds the random start, uniform in [0, pi]

    Minimises <sum w ZZ> in raw weights with mixer e^{-i beta X}, the same
    setup as upstream's depth-matched study, then converts to our convention.

    Return: (gammas, betas).
    """

    def ours(theta):
        return np.asarray(theta[:p]) * P.scale, -np.asarray(theta[p:])

    np.random.seed(seed)
    x0 = np.random.uniform(0, np.pi, 2 * p)
    res = minimize(
        lambda th: P.energy(*ours(th)), x0, method="COBYLA", options={"maxiter": max(150, 10 * p)}
    )
    return ours(res.x)
