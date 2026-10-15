"""LR-QAOA against COBYLA-optimized QAOA at the same depth.

Each cell (n, family, seed, p): COBYLA from a random start (np.random.seed(seed),
angles uniform in [0, pi], maxiter max(150, 10 p)) minimizes <sum w ZZ> with the
raw weights and mixer e^{-i beta X}. Both circuits are then scored on the exact
final state. "LR wins" means the ramp's P(x*) is higher.

n in {8, 10, 12} x 7 families x seeds 0-4 x p in {10, 20} = 210 cells.

    python experiments/depth_matched.py        # -> results/depth_matched.csv
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from maxcut_qiskit import worker_pool
from maxcut_qiskit.graphs import MIN_N, edge_arrays, make_graph

OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "depth_matched.csv"
FAMILIES = ["ring", "ramanujan", "clustered", "bipartite", "erdos_renyi", "euclidean", "dense"]


def optimize(P, p, seed):
    """optimize() - COBYLA angles for one instance, on the exact state
    @P: the Problem
    @p: number of layers
    @seed: seeds the random start

    Return: (gammas, betas, scipy result, number of evaluations), angles in the
    circuit's convention.
    """

    def native(theta):
        # optimizer works in raw weights with mixer rx(2 beta) = e^{-i beta X}
        return np.asarray(theta[:p]) * P.scale, -np.asarray(theta[p:])

    calls = [0]

    def cost(theta):
        calls[0] += 1
        return P.energy(*native(theta))

    np.random.seed(seed)
    x0 = np.random.uniform(0, np.pi, 2 * p)
    res = minimize(cost, x0, method="COBYLA", options={"maxiter": max(150, 5 * 2 * p)})
    return *native(res.x), res, calls[0]


def cell(args):
    from maxcut_qiskit.qaoa import Problem

    n, family, seed, p = args
    P = Problem(*edge_arrays(make_graph(family, n, seed)), n=n)
    P.optimum()
    g, b, res, calls = optimize(P, p, seed)
    opt = P.evaluate(g, b)
    lr = P.lr(p)
    return dict(
        n_nodes=n,
        family=family,
        seed=seed,
        p=p,
        ground_truth=P.best,
        cobyla_success_probability=opt["success_probability"],
        cobyla_mean_ratio=opt["mean_ratio"],
        cobyla_final_energy=float(res.fun),
        cobyla_evaluations=calls,
        lr_success_probability=lr["success_probability"],
        lr_mean_ratio=lr["mean_ratio"],
        backend=P.describe(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", nargs="*", type=int, default=[8, 10, 12])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--p", nargs="*", type=int, default=[10, 20])
    ap.add_argument("--families", nargs="*", default=FAMILIES)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    tasks = [
        (n, f, s, p)
        for n in a.n
        for f in a.families
        for s in a.seeds
        for p in a.p
        if n >= MIN_N.get(f, 0)
    ]
    with worker_pool(a.jobs, max_tasks=10) as pool:
        rows = pool.map(cell, tasks, chunksize=1)
    d = pd.DataFrame(rows).sort_values(["n_nodes", "family", "seed", "p"])
    OUT.parent.mkdir(exist_ok=True)
    d.to_csv(OUT, index=False)
    win = d.lr_success_probability > d.cobyla_success_probability
    print(
        f"LR-QAOA P(x*) above COBYLA in {int(win.sum())} of {len(d)} cells; "
        f"mean ratio above in {int((d.lr_mean_ratio > d.cobyla_mean_ratio).sum())} of {len(d)}"
    )
    print(d.assign(lr_wins=win).groupby(["family", "p"]).lr_wins.sum().unstack().to_string())
    print("wrote", OUT)


if __name__ == "__main__":
    main()
