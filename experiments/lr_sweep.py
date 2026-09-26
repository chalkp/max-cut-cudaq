"""LR-QAOA depth sweep against size, and the eta(p) fit.

Grid: n = 8..28 even, 7 families, p = 5..300 (16 depths), 12 seeds, Delta = (0.6, 0.3).
Ring and bipartite don't depend on the seed and only run seed 0. P(x*) and the
mean cut ratio are read off the state vector, no shots.

n is the outer loop, so stopping early still leaves every finished size complete.
The CSV is the checkpoint; rerun to resume. --hours counts from the first launch
and is kept in <out>.deadline, delete that file to start a fresh budget.

    python experiments/lr_sweep.py --hours 3
    python experiments/lr_sweep.py --fit           # -> results/lr_p_sweep_eta.csv
"""

import argparse
import csv
import pathlib
import subprocess
import sys
import time

import numpy as np

from maxcut_cudaq.graphs import MIN_N, edge_arrays, make_graph

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
N_LIST = list(range(8, 29, 2))
P_LIST = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300]
# cheap families first
FAMILIES = ["ring", "ramanujan", "clustered", "bipartite", "erdos_renyi", "euclidean", "dense"]
SEEDS = list(range(12))
DETERMINISTIC = {"ring", "bipartite"}
DG, DB = 0.6, 0.3
FP32_MIN_N = 22  # fp64 gets several times slower past ~20 qubits on GeForce

FIELDS = [
    "n_nodes",
    "family",
    "seed",
    "p",
    "n_edges",
    "ground_truth",
    "success_probability",
    "mean_ratio",
    "random_success_probability",
    "precision",
    "wall_clock_s",
    "delta_gamma",
    "delta_beta",
]


def deadline_for(out, hours):
    f = pathlib.Path(str(out) + ".deadline")
    if f.exists():
        return float(f.read_text().strip())
    d = time.time() + hours * 3600.0
    f.write_text(str(d))
    return d


def load_done(out):
    if not pathlib.Path(out).exists():
        return set()
    with open(out) as fh:
        return {
            (int(r["n_nodes"]), r["family"], int(r["seed"]), int(r["p"]))
            for r in csv.DictReader(fh)
        }


def worker(args):
    from maxcut_cudaq import qaoa

    deadline = deadline_for(args.out, args.hours)
    done = load_done(args.out)
    print(
        f"[sweep] {len(done)} runs done; {(deadline - time.time()) / 3600:.2f} h left", flush=True
    )
    new = not pathlib.Path(args.out).exists()
    fh = open(args.out, "a", newline="")
    wr = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        wr.writeheader()
    completed = 0
    for n in args.n:
        qaoa.set_precision("fp32" if n >= args.fp32_min_n else "fp64")
        for family in args.families:
            if n < MIN_N.get(family, 0):
                continue
            for seed in [0] if family in DETERMINISTIC else args.seeds:
                todo = [p for p in args.p if (n, family, seed, p) not in done]
                if not todo:
                    continue
                G = make_graph(family, n, seed)
                P = qaoa.Problem(*edge_arrays(G), n=n)
                optimum = P.optimum()
                print(
                    f"== n={n} {family} seed={seed} edges={G.number_of_edges()} optimum={optimum:g} "
                    f"[{qaoa.precision()}]",
                    flush=True,
                )
                for p in todo:
                    if time.time() >= deadline:
                        print("[budget] wall-clock budget reached.\nSWEEP COMPLETE", flush=True)
                        return 0
                    t0 = time.time()
                    r = P.lr(p, DG, DB)
                    dt = time.time() - t0
                    wr.writerow(
                        dict(
                            n_nodes=n,
                            family=family,
                            seed=seed,
                            p=p,
                            n_edges=G.number_of_edges(),
                            ground_truth=optimum,
                            success_probability=r["success_probability"],
                            mean_ratio=r["mean_ratio"],
                            random_success_probability=2.0 / 2**n,
                            precision=qaoa.precision(),
                            wall_clock_s=dt,
                            delta_gamma=DG,
                            delta_beta=DB,
                        )
                    )
                    fh.flush()
                    completed += 1
                    print(
                        f"  p={p:4d}: P(x*)={r['success_probability']:.5f} r={r['mean_ratio']:.4f} {dt:7.2f}s",
                        flush=True,
                    )
                    if completed >= args.max_runs:
                        print(f"[batch] {completed} runs; recycling the process.", flush=True)
                        return 7
                del P
    print(
        f"[sweep] n={args.n[0]} done" if len(args.n) == 1 else "SWEEP COMPLETE (grid exhausted)",
        flush=True,
    )
    return 0


MIN_SEEDS = 6  # sizes with fewer seeds stay out of a random family's fit


def fit_eta(d):
    """fit_eta() - fit log2 P(x*) = -eta n + C for every (family, p)
    @d: sweep rows

    Fits the log of the seed mean. Small n sits at P ~ 1 for deep circuits,
    which drags eta down at large p; check r2 before trusting it there.

    Return: DataFrame with eta, C, r2, sizes, dropped_thin and n_max.
    """
    import pandas as pd

    rows = []
    for (fam, p), g in d.groupby(["family", "p"]):
        counts = g.groupby("n_nodes")["success_probability"].count()
        m = g.groupby("n_nodes")["success_probability"].mean()
        need = 1 if fam in DETERMINISTIC else MIN_SEEDS
        thin = int((counts < need).sum())
        m = m[(counts >= need) & (m > 0)]
        if len(m) < 3:
            rows.append(
                dict(
                    family=fam,
                    p=p,
                    eta=np.nan,
                    C=np.nan,
                    r2=np.nan,
                    sizes=len(m),
                    dropped_thin=thin,
                    n_max=int(m.index.max()) if len(m) else 0,
                )
            )
            continue
        x, y = m.index.values.astype(float), np.log2(m.values)
        slope, icept = np.polyfit(x, y, 1)
        ss_res, ss_tot = (
            float(((y - slope * x - icept) ** 2).sum()),
            float(((y - y.mean()) ** 2).sum()),
        )
        rows.append(
            dict(
                family=fam,
                p=p,
                eta=-slope,
                C=icept,
                r2=1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
                sizes=len(m),
                dropped_thin=thin,
                n_max=int(m.index.max()),
            )
        )
    return pd.DataFrame(rows).sort_values(["family", "p"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument(
        "--max-runs",
        type=int,
        default=400,
        help="worker exits 7 after this many runs; the supervisor restarts it",
    )
    ap.add_argument("--out", default=str(RESULTS / "lr_p_sweep.csv"))
    ap.add_argument("--n", nargs="*", type=int, default=N_LIST)
    ap.add_argument("--p", nargs="*", type=int, default=P_LIST)
    ap.add_argument("--families", nargs="*", default=FAMILIES)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--fp32-min-n", type=int, default=FP32_MIN_N)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--fit", action="store_true", help="fit eta(p) on the CSV and exit")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if a.fit:
        import pandas as pd

        fits = fit_eta(pd.read_csv(a.out))
        path = pathlib.Path(a.out).with_name(pathlib.Path(a.out).stem + "_eta.csv")
        fits.to_csv(path, index=False)
        print(fits.round(4).to_string(index=False), f"\nwrote {path}")
        return 0
    if a.worker:
        return worker(a)
    # A long-lived cudaq process keeps growing on the host, so the worker
    # exits 7 every --max-runs runs and gets restarted.
    passthrough = [
        "--hours",
        str(a.hours),
        "--max-runs",
        str(a.max_runs),
        "--out",
        a.out,
        "--fp32-min-n",
        str(a.fp32_min_n),
        "--p",
        *map(str, a.p),
        "--families",
        *a.families,
        "--seeds",
        *map(str, a.seeds),
    ]
    for n in a.n:
        while True:
            code = subprocess.call(
                [sys.executable, __file__, "--worker", "--n", str(n), *passthrough]
            )
            if code != 7:
                break
        if code != 0:
            return code
        if time.time() >= deadline_for(a.out, a.hours):
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
