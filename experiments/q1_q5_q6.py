"""Numbers for open questions Q1, Q5 and Q6. Brute force goes through qaoa.Problem,
the rest is closed form.

Q1  Ring, speed ~ gap^q: how does the depth for P = 1/2 grow with n?
    n up to 2048, q = 1..2; is the exponent max(1, q - 1/2)? Does the q > 3/2
    penalty come from the ends (a taper should kill it)? Quasi-energy gap
    instead of the continuum one. Which modes carry the loss.

Q5  How should Delta shrink with n on dense graphs?
    K(n/2, n/2) and K_n exactly at any n: best Delta and where more depth
    starts to hurt (dg = db). Weighted K_n by brute force up to n = 18, positive
    weights (as in the hardware paper) and signed ones (our dense family).

Q6  Is the "depth hurts" threshold dg = pi / h_max, h_max the largest local
    field in a best cut?

    python experiments/q1_q5_q6.py             # adds keys to results/q5_q1_numbers.json
    python experiments/q1_q5_q6.py --only q1   # or q5, q6
"""

import argparse
import json
import pathlib

import numpy as np

from maxcut_cudaq import worker_pool
from maxcut_cudaq.exact import (
    GapPower,
    complete_success,
    bipartite_success,
    first_depth,
    mode_probabilities,
    momenta,
    ring_log2_success,
)
from maxcut_cudaq.graphs import edge_arrays, special_graph, weighted_complete

OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "q5_q1_numbers.json"
DG, DB = 0.6, 0.3
MAX_TASKS = 200  # cudaq workers grow on the host, recycle them


# =============================================================================== Q1
def depth_row(args):
    label, q, kind, taper, sizes = args
    out, start = [], 2
    for n in sizes:
        sched = GapPower(n, q, kind, taper)
        p = first_depth(n, sched, start=start)
        # warm start from the last size; back off if we started past the answer
        while start > 2 and p == start:
            start = max(2, start // 2)
            p = first_depth(n, sched, start=start)
        out.append(p)
        start = max(2, int(p * 0.9))
    return label, q, out


def loss_split(n, q, p, k_cut=0.3):
    """Fraction of -log2 P carried by modes with k > k_cut."""
    k = momenta(n)
    u = GapPower(n, q)(p)
    Pk = mode_probabilities(k, u[1:] * DG, (1 - u[:-1]) * DB)
    loss = -np.log2(Pk)
    return float(loss[k > k_cut].sum() / loss.sum()), float(loss.sum())


def run_q1(R, jobs):
    sizes = [64, 128, 256, 512, 1024, 2048]
    small = sizes[:5]
    tasks = [
        (f"q={q}", q, "continuum", False, sizes if q <= 1.6 else small)
        for q in (1.0, 1.25, 1.4, 1.5, 1.6, 1.75, 2.0)
    ]
    tasks += [(f"q={q} tapered", q, "continuum", True, small) for q in (1.5, 1.75, 2.0)]
    tasks += [(f"q={q} quasi-energy gap", q, "floquet", False, small) for q in (1.5, 2.0)]
    tasks += [(f"q={q} tapered", q, "continuum", True, sizes) for q in (1.9, 2.1, 2.25, 2.5)]
    tasks += [("q=2.0 tapered, n to 2048", 2.0, "continuum", True, sizes)]
    tasks += [(f"q={q} smoother taper", q, "continuum", "smoother", small) for q in (2.0, 2.5)]
    tasks += [("q=2.0 tapered, quasi-energy", 2.0, "floquet", True, small)]
    with worker_pool(jobs, MAX_TASKS) as pool:
        rows = pool.map(depth_row, tasks)
    print("Q1. layers for P(x*) >= 1/2 on the ring (first depth on a 5%-spaced grid)")
    print(
        f"   {'schedule':26s}"
        + "".join(f"{n:>8d}" for n in sizes)
        + "   local exp   predicted   layers/n (last two)"
    )
    table = {}
    for label, q, ps in rows:
        ns = sizes[: len(ps)]
        a = float(np.polyfit(np.log(ns[-3:]), np.log(ps[-3:]), 1)[0])
        pred = (2 - q) if q < 1 else (1.0 if q == 1 else max(1.0, q - 0.5))
        table[label] = dict(
            q=q,
            n=ns,
            p_half=ps,
            local_exponent=a,
            predicted=pred,
            layers_per_n=[p / n for p, n in zip(ps, ns)],
        )
        print(
            f"   {label:26s}"
            + "".join(f"{p:8d}" for p in ps)
            + " " * (8 * (len(sizes) - len(ps)))
            + f"   {a:8.2f}   {pred:9.2f}   {ps[-2] / ns[-2]:.2f}, {ps[-1] / ns[-1]:.2f}"
        )
    R["q1_depths"] = table

    print(
        "   share of layers in the outer tenths of u, and share of the loss in modes with k > 0.3, at n = 512:"
    )
    ends = {}
    for q in (1.0, 1.5, 1.75, 2.0):
        ph = table[f"q={q}"]["p_half"][3]
        frac, total = loss_split(512, q, ph)
        ends[q] = dict(
            end_fraction=GapPower(512, q).end_fraction(), high_mode_loss_share=frac, p=ph
        )
        print(
            f"     q={q}: {ends[q]['end_fraction']:.2e} of layers at the ends; {frac:.2f} of the loss from k > 0.3 (at p = {ph})"
        )
    R["q1_ends"] = ends

    print("   success probability against depth for the tapered q = 2 schedule:")
    curve = {}
    for n in (100, 512, 2048):
        sched = GapPower(n, 2.0, taper=True)
        curve[n] = {
            r: 2.0 ** ring_log2_success(n, int(r * n), DG, DB, u=sched(int(r * n)))
            for r in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0)
        }
        print(f"     n={n:5d}: " + "  ".join(f"p={r}n: {v:.3f}" for r, v in curve[n].items()))
    R["q1_tapered_curve"] = {str(n): {str(r): v for r, v in c.items()} for n, c in curve.items()}
    n = 100
    lin = first_depth(n, lambda p: np.linspace(0, 1, p + 1))
    print(
        f"   at n = 100: linear ramp {lin}, q=1.5 {first_depth(n, GapPower(n, 1.5))}, "
        f"tapered q=2 {first_depth(n, GapPower(n, 2.0, taper=True))}, variational optimum 50"
    )


# =============================================================================== Q5
def delta_profile(fn, n, grid, depths=(10, 50, 200)):
    """Best common Delta per depth, and the cliff past which depth 200 loses to depth 50."""
    P = {p: np.array([fn(n, p, d, d) for d in grid]) for p in depths}
    best = {p: float(grid[int(np.argmax(P[p]))]) for p in depths}
    ok = P[depths[-1]] >= P[depths[-2]]
    i0 = int(np.argmax(P[depths[-1]]))
    bad = [i for i in range(i0, len(grid)) if not ok[i]]
    cliff = float(grid[bad[0]]) if bad else float("nan")
    return dict(n=n, best=best, best_P={p: float(P[p].max()) for p in depths}, cliff=cliff)


GRID_FULL = np.round(np.geomspace(0.05, 1.5, 22), 4)


def brute_task(args):
    from maxcut_cudaq.qaoa import Problem

    n, seed, signed = args
    P = Problem(*weighted_complete(n, seed, "signed" if signed else "positive"), n=n)
    rows = []
    for p in (10, 50):
        for d in GRID_FULL:
            m = P.lr(p, d, d)
            rows.append(
                (
                    n,
                    seed,
                    signed,
                    p,
                    float(d),
                    m["success_probability"],
                    m["mean_ratio"],
                    P.describe(),
                )
            )
    return rows


def run_q5(R, jobs, n_max):
    print("Q5. exact collective-spin models, Delta_gamma = Delta_beta = Delta")
    grid = np.geomspace(0.005, 1.5, 140)
    out = {}
    for name, fn, sizes in (
        ("complete bipartite K(n/2,n/2)", bipartite_success, (16, 32, 64, 128)),
        ("complete graph K(n)", complete_success, (16, 32, 64, 128, 256)),
    ):
        with worker_pool(jobs, MAX_TASKS) as pool:
            rows = pool.starmap(delta_profile, [(fn, n, grid) for n in sizes])
        out[name] = rows
        for r in rows:
            print(
                f"   {name:30s} n={r['n']:4d}: best Delta at p=10/50/200 = "
                + "/".join(f"{r['best'][p]:.3f}" for p in (10, 50, 200))
                + f"  (x n = {r['best'][200] * r['n']:.2f});  P at best = "
                + "/".join(f"{r['best_P'][p]:.2f}" for p in (10, 50, 200))
                + f";  depth starts to hurt at Delta = {r['cliff']:.3f} (x n = {r['cliff'] * r['n']:.2f})"
            )
        ns = np.array([r["n"] for r in rows])
        b = np.array([r["best"][200] for r in rows])
        c = np.array([r["cliff"] for r in rows])
        print(
            f"      scaling: best Delta ~ n^{np.polyfit(np.log(ns), np.log(b), 1)[0]:+.2f},  cliff ~ n^{np.polyfit(np.log(ns), np.log(c), 1)[0]:+.2f}"
        )
    R["q5_collective"] = out

    print(
        f"   weighted complete graphs by brute force, n <= {n_max}, 5 seeds, Delta_gamma = Delta_beta"
    )
    tasks = [
        (n, s, signed) for n in range(8, n_max + 1, 2) for s in range(5) for signed in (False, True)
    ]
    with worker_pool(jobs, MAX_TASKS) as pool:
        rows = [r for part in pool.imap_unordered(brute_task, tasks, chunksize=1) for r in part]
    import pandas as pd

    d = pd.DataFrame(rows, columns=["n", "seed", "signed", "p", "delta", "ps", "ratio", "backend"])
    summary = {}
    for signed, name in (
        (False, "positive weights {0.1,0.2,0.3,0.5,1}"),
        (True, "signed weights -10..10"),
    ):
        g = (
            d[d.signed == signed]
            .groupby(["n", "p", "delta"])
            .agg(ps=("ps", "mean"), ratio=("ratio", "mean"))
            .reset_index()
        )
        print(f"   {name}")
        fam = []
        for p in (10, 50):
            line = []
            for n in sorted(g.n.unique()):
                x = g[(g.n == n) & (g.p == p)]
                bp, br = x.loc[x.ps.idxmax()], x.loc[x.ratio.idxmax()]
                fam.append(
                    dict(
                        n=int(n),
                        p=p,
                        best_delta_ps=float(bp.delta),
                        ps=float(bp.ps),
                        best_delta_ratio=float(br.delta),
                        ratio=float(br.ratio),
                    )
                )
                line.append(
                    f"n={n}: {bp.delta:.2f} (P={bp.ps:.2f}) / {br.delta:.2f} (r={br.ratio:.3f})"
                )
            print(f"      p={p:3d}, best Delta by P(x*) / by mean ratio:  " + ";  ".join(line))
            ns = np.array([f["n"] for f in fam if f["p"] == p])
            bd = np.array([f["best_delta_ratio"] for f in fam if f["p"] == p])
            print(
                f"              best Delta (by mean ratio) ~ n^{np.polyfit(np.log(ns), np.log(bd), 1)[0]:+.2f}"
            )
        summary[name] = fam
    R["q5_brute"] = summary
    d.to_csv(OUT.parent / "q5_delta_scan_complete.csv", index=False)


# =============================================================================== Q6
def local_field_max(P):
    """Largest local field over the best cuts (flipping i out of one raises H_C by 2 h_i)."""
    n = P.n
    hc = P.hc_host()
    opt = np.where(P.optimal_mask_host())[0]
    return max(float((hc[opt ^ (1 << i)] - hc[opt]).max()) / 2 for i in range(n)), int(opt.size)


def threshold_task(args):
    """P(400 layers) / P(50 layers) around the predicted pi / h_max, db = 0.3."""
    from maxcut_cudaq.qaoa import Problem

    kind, n, seed = args
    G = special_graph(kind, n, seed)
    P = Problem(*edge_arrays(G), n=G.number_of_nodes())
    h_max, n_opt = local_field_max(P)
    pred = np.pi / h_max
    factors = (0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5)
    row = dict(
        graph=kind,
        n=P.n,
        seed=seed,
        n_optimal=n_opt,
        h_max=h_max,
        predicted=pred,
        factors=list(factors),
        shallow=[],
        deep=[],
        backend=P.describe(),
    )
    for fct in factors:
        for p, key in ((50, "shallow"), (400, "deep")):
            row[key].append(P.lr(p, fct * pred, 0.3)["success_probability"])
    return row


def run_q6(R, jobs):
    tasks = [
        ("ring", 12, 0),
        ("hypercube", 8, 0),
        ("hypercube", 16, 0),
        ("torus 4x4", 16, 0),
        ("complete bipartite", 12, 0),
        ("Petersen", 10, 0),
        ("ramanujan", 14, 0),
        ("ramanujan", 14, 1),
        ("erdos_renyi", 12, 0),
        ("clustered", 12, 0),
        ("euclidean", 12, 0),
        ("dense", 12, 0),
        ("dense", 14, 1),
        ("heterogeneous_weight", 12, 0),
    ]
    with worker_pool(min(jobs, len(tasks)), MAX_TASKS) as pool:
        rows = pool.map(threshold_task, tasks, chunksize=1)
    print(
        'Q6. is the "depth hurts" threshold Delta_gamma = pi / (largest local field in the best cut)?  Delta_beta = 0.3'
    )
    print("    P(x*) at p = 400 divided by P(x*) at p = 50, at Delta_gamma = factor x prediction")
    print(
        f"    {'graph':22s} {'n':>3s} {'h_max':>6s} {'pred.':>6s} "
        + "".join(f"{f:>9.2f}" for f in rows[0]["factors"])
    )
    for r in rows:
        ratio = [d / max(s_, 1e-12) for d, s_ in zip(r["deep"], r["shallow"])]
        r["ratio"] = ratio
        print(
            f"    {r['graph']:22s} {r['n']:3d} {r['h_max']:6.2f} {r['predicted']:6.3f} "
            + "".join(f"{x:9.2f}" for x in ratio)
            + f"   deep P at 0.9/1.1: {r['deep'][2]:.2f}/{r['deep'][4]:.3f}"
        )
    R["q6_threshold"] = rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["q1", "q5", "q6"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--n-max", type=int, default=18)
    a = ap.parse_args()
    OUT.parent.mkdir(exist_ok=True)
    R = json.loads(OUT.read_text()) if OUT.exists() else {}
    if a.only in (None, "q1"):
        run_q1(R, a.jobs)
    if a.only in (None, "q5"):
        run_q5(R, a.jobs, a.n_max)
    if a.only in (None, "q6"):
        run_q6(R, a.jobs)
    OUT.write_text(json.dumps(R, indent=1))
    print("wrote", OUT)
