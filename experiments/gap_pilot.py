"""Gap pilot, n <= 14: where is the gap along the ramp, and does slowing down there
beat the linear ramp at the same depth?

The continuous-time path behind the ramp is H(u) = (1 - u) db H_M + u dg H_C with
H_M = -sum X and u the layer fraction. Both terms keep the global flip prod X, and
|+>^n is even under it, so everything lives in the even sector. With m = (number
of optimal bitstrings) / 2, the lowest m levels there all end up optimal and
leaking between them is free. The gap that matters is the one to the lowest
level j >= m with V_j0 != 0, V = dH/du.

Schedules, same endpoints and Delta, same depth:
  linear        u(t) = t
  adiabatic     du/dt ~ A(u)^(-1/2), A = sum_{j>=m} |V_j0|^2 / (E_j - E_0)^2.
                Needs the spectrum, so it's an oracle: an upper bound on what
                knowing the gap buys, not something you'd run on hardware.
  slow_middle   u(t) = (1 + (2t - 1)^3) / 2
  slow_ends     u(t) = 3t^2 - 2t^3

    python experiments/gap_pilot.py            # -> results/gap_pilot.csv
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from maxcut_qiskit import worker_pool
from maxcut_qiskit.graphs import MIN_N, edge_arrays, make_graph

DG, DB = 0.6, 0.3
U_GRID = np.linspace(0.0, 1.0, 51)
OUTPUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "gap_pilot.csv"
FAMILIES = [
    "ring",
    "bipartite",
    "ramanujan",
    "irregular",
    "clustered",
    "erdos_renyi",
    "euclidean",
    "heterogeneous_weight",
    "dense",
]


def instance(family, n, seed):
    from maxcut_qiskit.qaoa import Problem

    G = make_graph(family, n, seed)
    return Problem(*edge_arrays(G), n=G.number_of_nodes())


def even_sector(n, hc):
    """H_M and H_C on the +1 sector of prod X, basis (|x> + |~x>) / sqrt2 with the top bit 0."""
    N, half = 2**n, 2 ** (n - 1)
    x = np.arange(half)
    rows, cols = [], []
    for j in range(n):
        y = x ^ (1 << j)
        y = np.where(y >= half, (~y) & (N - 1), y)  # fold back to the top-bit-0 twin
        rows.append(x)
        cols.append(y)
    HM = -sp.coo_matrix(
        (np.ones(n * half), (np.concatenate(rows), np.concatenate(cols))), shape=(half, half)
    ).tocsr()
    return HM, hc[:half]


def spectrum_along_path(HM, hc_even, m, k_extra=10):
    """spectrum_along_path() - relevant gap and leak weight A(u) on U_GRID
    @HM: mixer on the even sector
    @hc_even: diagonal of H_C on the even sector
    @m: number of levels that end in the optimal manifold
    @k_extra: eigenpairs computed beyond m

    XXX: near u = 0 the first excited level is hugely degenerate and only
    m + k_extra pairs are computed, so eigsh's random start vector decides
    which part of the multiplet ends up in A. The adiabatic arm is not
    reproducible run to run there. The gap itself is. The same random start
    now and then keeps eigsh from converging; it is retried from a fresh one,
    since the error does not survive the trip back from a pool worker.
    """
    half = hc_even.size
    k = min(m + k_extra, half - 2)
    V = (-DB) * HM + DG * sp.diags(hc_even)
    gap, A = np.full(U_GRID.size, np.nan), np.full(U_GRID.size, np.nan)
    v0 = None
    for i, u in enumerate(U_GRID):
        H = (1 - u) * DB * HM + u * DG * sp.diags(hc_even)
        for attempt in range(3):
            try:
                E, W = spla.eigsh(H, k=k, which="SA", v0=v0, tol=1e-9, maxiter=20000)
                break
            except spla.ArpackNoConvergence:
                if attempt == 2:
                    raise RuntimeError(f"eigsh did not converge at u = {u}") from None
                v0 = None
        o = np.argsort(E)
        E, W = E[o], W[:, o]
        v0 = W[:, 0]
        c = np.abs(W[:, m:].T @ (V @ W[:, 0]))
        dE = E[m:] - E[0]
        ok = (c > 1e-7) & (dE > 1e-10)
        if ok.any():
            gap[i] = dE[ok].min()
            A[i] = np.sum(c[ok] ** 2 / dE[ok] ** 2)
    return gap, A


def schedule(name, p, A=None):
    """u at the layer boundaries t = 0, 1/p, ..., 1."""
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


def run_instance(args):
    family, n, seed, depths = args
    P = instance(family, n, seed)
    hc = P.hc_host()
    opt = P.optimal_mask_host()
    m = int(opt.sum()) // 2
    HM, hc_even = even_sector(n, hc)
    gap, A = spectrum_along_path(HM, hc_even, m)
    i_min = int(np.nanargmin(gap))
    rows = []
    for p in depths:
        for name in ("linear", "adiabatic", "slow_middle", "slow_ends"):
            r = P.lr(p, DG, DB, u=schedule(name, p, A))
            rows.append(
                dict(
                    n_nodes=n,
                    family=family,
                    seed=seed,
                    p=p,
                    schedule=name,
                    success_probability=r["success_probability"],
                    mean_ratio=r["mean_ratio"],
                    n_optimal=int(opt.sum()),
                    gap_min=float(gap[i_min]),
                    u_at_gap_min=float(U_GRID[i_min]),
                    gap_end=float(gap[-1]) if np.isfinite(gap[-1]) else float("nan"),
                    backend=P.describe(),
                )
            )
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", nargs="*", type=int, default=[8, 10, 12, 14])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--depths", nargs="*", type=int, default=[10, 20, 50, 100])
    ap.add_argument("--families", nargs="*", default=FAMILIES)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    tasks = [
        (f, n, s, args.depths)
        for n in args.n
        for f in args.families
        for s in args.seeds
        if n >= MIN_N.get(f, 0)
    ]
    with worker_pool(args.jobs, max_tasks=20) as pool:
        out = [r for rows in pool.imap_unordered(run_instance, tasks) for r in rows]
    OUTPUT.parent.mkdir(exist_ok=True)
    pd.DataFrame(out).sort_values(["n_nodes", "family", "seed", "p", "schedule"]).to_csv(
        OUTPUT, index=False
    )
    print(f"wrote {len(out)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
