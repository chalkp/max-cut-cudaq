"""Follow-up to q1_q5_q6.py: which constraint limits dg, and at what depth.

Three candidate limits on the cost step of a linear ramp:
  k = 1       dg * 2 h_max = 2 pi   a single flip out of the best cut wraps around
  k = 2       dg * dE_2 = 2 pi      same for a two-vertex flip; second order, so it
                                    only shows up deep or at a big mixer step
  collective  dg * lambda * max_u[u tan((1 - u) db)] = 1, lambda = n * mean weight;
              the uniform mode period-doubles mid-ramp (exact for K_n at large n)

Runs:
  kn      K_n, one-spin model, n <= 512: is the cliff at the collective limit?
  deep    ten graphs at 400 / 1600 / 6400 layers just below the k = 1 limit
  dest    where the lost probability goes (Hamming distance, energy)
  beta    weighted K_n, n = 14 and 16: does the cliff move with db?
  bip     K(n/2, n/2), two-spin model, at a large mixer step
  field   how h_max grows with n on weighted K_n

    python experiments/delta_limits.py       # adds keys to results/q5_q1_numbers.json
"""

import argparse
import json
import pathlib

import numpy as np
from scipy.optimize import minimize_scalar

from maxcut_cudaq import worker_pool
from maxcut_cudaq.exact import bipartite_state, complete_run
from maxcut_cudaq.graphs import edge_arrays, special_graph
from maxcut_cudaq.graphs import weighted_complete as weighted_complete_graph

OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "q5_q1_numbers.json"


def midramp(db):
    """max_u u tan((1 - u) db), and the u where it sits."""
    r = minimize_scalar(lambda u: -u * np.tan((1 - u) * db), bounds=(0, 1), method="bounded")
    return -r.fun, r.x


# =============================================================================== complete graph
KN_FACTORS = (0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0)


def kn_task(args):
    n, db = args
    mx, ustar = midramp(db)
    dgc = 1.0 / (n * mx)
    row = dict(
        n=n,
        db=db,
        predicted=dgc,
        u_at_threshold=ustar,
        factors=list(KN_FACTORS),
        P400=[],
        P1600=[],
        sz2_400=[],
        sz2_1600=[],
    )
    for f in KN_FACTORS:
        for p in (400, 1600):
            P, s = (None, None) if (n == 512 and p == 1600) else complete_run(n, p, f * dgc, db)
            row[f"P{p}"].append(P)
            row[f"sz2_{p}"].append(s)
    return row


# =============================================================================== graphs by brute force
def instance_arrays(P):
    """(x, H_C, optimal indices, h_max, mean normalized weight)."""
    hc = P.hc_host()
    opt = np.where(P.optimal_mask_host())[0]
    hmax = max(float((hc[opt ^ (1 << i)] - hc[opt]).max()) / 2 for i in range(P.n))
    return np.arange(1 << P.n), hc, opt, hmax, float(P.wn.mean())


def graph_problem(kind, n, seed):
    from maxcut_cudaq.qaoa import Problem

    G = special_graph(kind, n, seed)
    return Problem(*edge_arrays(G), n=G.number_of_nodes())


def complete_problem(n, seed, kind):
    from maxcut_cudaq.qaoa import Problem

    return Problem(*weighted_complete_graph(n, seed, kind), n=n)


def destination(x, hc, opt, pr, dg, top=4):
    """Lost probability by (Hamming distance to the nearest best cut, energy above it)."""
    ham = np.min(np.array([np.bitwise_count(x ^ o) for o in opt]), axis=0)
    dE = np.round(hc - hc[opt[0]], 6)
    groups = {}
    for d, e, q in zip(ham[ham > 0], dE[ham > 0], pr[ham > 0]):
        groups[(int(d), float(e))] = groups.get((int(d), float(e)), 0.0) + float(q)
    best = sorted(groups.items(), key=lambda kv: -kv[1])[:top]
    return (
        dict((d, float(pr[ham == d].sum())) for d in range(1, 8)),
        [dict(hamming=d, dE=e, prob=q, wraps=float(dg * e / (2 * np.pi))) for (d, e), q in best],
    )


def deep_task(args):
    kind, n, seed = args
    P = graph_problem(kind, n, seed)
    x, hc, opt, hmax, _ = instance_arrays(P)
    pred = np.pi / hmax
    row = dict(graph=kind, n=P.n, seed=seed, h_max=hmax, predicted=pred, P={}, backend=P.describe())
    for f in (0.55, 0.7, 0.8, 0.9, 0.95):
        row["P"][f] = [P.lr(p, f * pred, 0.3)["success_probability"] for p in (400, 1600, 6400)]
    return row


def dest_task(args):
    kind, n, seed, f, p = args
    P = graph_problem(kind, n, seed)
    x, hc, opt, hmax, _ = instance_arrays(P)
    dg = f * np.pi / hmax
    pr = P.lr_probabilities(p, dg, 0.3)
    by_h, top_groups = destination(x, hc, opt, pr, dg)
    return dict(
        graph=kind,
        n=P.n,
        f=f,
        p=p,
        h_max=hmax,
        P_opt=float(pr[opt].sum()),
        by_hamming=by_h,
        top=top_groups,
        backend=P.describe(),
    )


BETA_DG = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2)


def beta_task(args):
    n, seed, kind, db = args
    P = complete_problem(n, seed, kind)
    x, hc, opt, hmax, wbar = instance_arrays(P)
    row = dict(
        n=n,
        seed=seed,
        kind=kind,
        db=db,
        h_max=hmax,
        w_mean=wbar,
        local=np.pi / hmax,
        collective=1.0 / (n * wbar * midramp(db)[0]),
        dg=list(BETA_DG),
        P50=[],
        P400=[],
        backend=P.describe(),
    )
    for dg in BETA_DG:
        row["P50"].append(P.lr(50, dg, db)["success_probability"])
        row["P400"].append(P.lr(400, dg, db)["success_probability"])
    return row


def half_max_cliff(dg, P):
    """dg where P drops to half its peak, interpolated."""
    dg, P = np.asarray(dg), np.asarray(P)
    i0 = int(np.argmax(P))
    for i in range(i0, len(P) - 1):
        if P[i] >= P[i0] / 2 > P[i + 1]:
            return float(dg[i] + (dg[i + 1] - dg[i]) * (P[i] - P[i0] / 2) / (P[i] - P[i + 1]))
    return float("nan")


def dehn_comparison():
    """dehn_comparison() - Dehn et al. random-weight fit against the collective limit

    arXiv:2504.08577 App. D: dg = 5.945 N^-0.800, db = 1.13 N^-0.206. Their
    cost is sum_ij Q_ij z_i z_j, Q symmetric with off-diagonal <= 2, i.e. at
    most 1 per ZZ pair like ours. Mean weight 0.5 for the limit.
    """
    rows = []
    for N in (12, 16, 20, 25, 28, 30, 40, 56):
        dg, db = 5.945 * N**-0.800, 1.13 * N**-0.206
        rows.append(
            dict(
                N=N,
                dg=dg,
                db=db,
                product_times_N=dg * db * N,
                threshold=1 / (N * 0.5 * midramp(db)[0]),
                depth=0.066 * N**2.039,
            )
        )
    for N, dg, db in (
        (20, 0.542, 0.608),
        (25, 0.453, 0.581),
        (30, 0.392, 0.560),
    ):  # printed in the paper
        rows.append(
            dict(
                N=N,
                dg=dg,
                db=db,
                product_times_N=dg * db * N,
                threshold=1 / (N * 0.5 * midramp(db)[0]),
                printed=True,
            )
        )
    for r in rows:
        r["ratio"] = r["dg"] / r["threshold"]
    return rows


def beta_dest_task(args):
    n, seed, kind, db, dg = args
    P = complete_problem(n, seed, kind)
    x, hc, opt, hmax, wbar = instance_arrays(P)
    pr = P.lr_probabilities(400, dg, db)
    by_h, top_groups = destination(x, hc, opt, pr, dg)
    # 0 = as good as the best cut, 1 = no better than a random cut
    return dict(
        n=n,
        seed=seed,
        kind=kind,
        db=db,
        dg=dg,
        local=np.pi / hmax,
        P_opt=float(pr[opt].sum()),
        by_hamming=by_h,
        top=top_groups,
        backend=P.describe(),
        energy_vs_random=float((pr @ hc - hc[opt[0]]) / (hc.mean() - hc[opt[0]])),
    )


def field_task(args):
    """h_max of one weighted K_n from the Q5 brute-force set."""
    n, seed, signed = args
    P = complete_problem(n, seed, "signed" if signed else "positive")
    return n, signed, instance_arrays(P)[3]


# =============================================================================== complete bipartite
def bip_task(args):
    """P(x*) and the weight one flip, two flips across and two flips same-side away."""
    n, db, c = args
    out = dict(n=n, db=db, dgn=c)
    for p in (200, 800):
        q = bipartite_state(n, p, c / n, db)
        out[f"P{p}"] = float(q[0, -1] + q[-1, 0])
        out[f"one_flip_{p}"] = float(q[1, -1] + q[0, -2] + q[-2, 0] + q[-1, 1])
        out[f"two_flip_across_{p}"] = float(q[1, -2] + q[-2, 1])
        out[f"two_flip_same_{p}"] = float(q[2, -1] + q[0, -3] + q[-3, 0] + q[-1, 2])
    return out


MAX_TASKS = 50  # cudaq workers grow on the host, recycle them
BIP_GRID = (2.0, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0, 5.6, 6.0, 6.4, 7.0)
DEEP_GRAPHS = [
    ("ramanujan", 14, 0),
    ("ramanujan", 14, 1),
    ("hypercube", 16, 0),
    ("Petersen", 10, 0),
    ("hypercube", 8, 0),
    ("complete bipartite", 12, 0),
    ("dense", 12, 0),
    ("euclidean", 12, 0),
    ("erdos_renyi", 12, 0),
    ("clustered", 12, 0),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    fm = lambda v: "   n/a" if v is None else f"{v:6.3f}"
    OUT.parent.mkdir(exist_ok=True)
    with worker_pool(a.jobs, MAX_TASKS, executor=True) as ex:
        f_kn = [
            ex.submit(kn_task, (n, db))
            for db in (0.3, 0.6, 1.0)
            for n in (512, 256, 128, 64, 32, 16)
        ]
        f_deep = [ex.submit(deep_task, g) for g in DEEP_GRAPHS]
        f_dest = [
            ex.submit(dest_task, (*g, f, 6400))
            for g in (
                ("ramanujan", 14, 0),
                ("hypercube", 16, 0),
                ("complete bipartite", 12, 0),
                ("euclidean", 12, 0),
            )
            for f in (0.9, 0.8)
        ]
        f_beta = [
            ex.submit(beta_task, (n, s, kind, db))
            for kind in ("uniform", "hardware")
            for n in (14, 16)
            for s in (0, 1)
            for db in (0.3, 0.6, 1.0)
        ]
        f_bdest = [
            ex.submit(beta_dest_task, (14, s, "uniform", db, dg))
            for s in (0, 1)
            for db, dg in ((0.3, 1.0), (0.6, 0.7), (1.0, 0.5))
        ]
        f_bip = [
            ex.submit(bip_task, (n, db, c))
            for n in (32, 64)
            for db in (0.3, 1.0, 1.5)
            for c in BIP_GRID
        ]
        f_field = [
            ex.submit(field_task, (n, sd, signed))
            for n in range(8, 19, 2)
            for sd in range(5)
            for signed in (False, True)
        ]

        kn = sorted((f.result() for f in f_kn), key=lambda r: (r["db"], r["n"]))
        print(
            "kn. complete graph: Delta_gamma = factor x the collective threshold;  factors "
            + " ".join(f"{f:6.2f}" for f in KN_FACTORS)
        )
        for r in kn:
            print(
                f"  Delta_beta={r['db']} n={r['n']:4d}  threshold Delta_gamma*n = {r['predicted'] * r['n']:.2f} (product with Delta_beta {r['predicted'] * r['n'] * r['db']:.2f})"
            )
            for key, label in (
                ("P400", "P(x*), 400 layers  "),
                ("P1600", "P(x*), 1600 layers "),
                ("sz2_400", "S_z^2/random, 400  "),
                ("sz2_1600", "S_z^2/random, 1600 "),
            ):
                print(f"      {label}" + " ".join(fm(v) for v in r[key]))

        deep = [f.result() for f in f_deep]
        print(
            "deep. P(x*) at 400 / 1600 / 6400 layers, Delta_gamma = factor x pi/h_max, Delta_beta = 0.3"
        )
        for r in deep:
            print(
                f"  {r['graph']:20s} n={r['n']:2d} "
                + "   ".join(f"{f}x: " + "/".join(f"{v:.3f}" for v in r["P"][f]) for f in r["P"])
            )

        dest = [f.result() for f in f_dest]
        print("dest. lost probability at 6400 layers, Delta_beta = 0.3")
        for r in dest:
            print(
                f"  {r['graph']:18s} n={r['n']:2d} {r['f']}x  P(x*)={r['P_opt']:.3f}  by Hamming distance 1..5: "
                + " ".join(f"{r['by_hamming'][d]:.3f}" for d in range(1, 6))
            )
            for t in r["top"]:
                print(
                    f"        Hamming {t['hamming']}, energy +{t['dE']:.3f}: prob {t['prob']:.3f}   Delta_gamma*dE/2pi = {t['wraps']:.3f}"
                )

        bip = [f.result() for f in f_bip]
        print("bip. complete bipartite: P(x*) and where the rest goes, against Delta_gamma * n")
        for n in (32, 64):
            for db in (0.3, 1.0, 1.5):
                rr = [r for r in bip if r["n"] == n and r["db"] == db]
                print(
                    f"  n={n} Delta_beta={db}: collective {2 / midramp(db)[0]:.2f}, k=1 wrap {2 * np.pi:.2f}, k=2 wrap {2 * np.pi * n / (2 * n - 4):.2f}"
                )
                print("      Delta_gamma*n       " + " ".join(f"{r['dgn']:5.1f}" for r in rr))
                for key in (
                    "P200",
                    "P800",
                    "one_flip_800",
                    "two_flip_across_800",
                    "two_flip_same_800",
                ):
                    print(f"      {key:20s}" + " ".join(f"{r[key]:5.3f}" for r in rr))

        beta = [f.result() for f in f_beta]
        print(
            "beta. weighted complete graphs: P(x*) at 400 layers and its ratio to 50 layers;  Delta_gamma "
            + " ".join(f"{d:6.2f}" for d in BETA_DG)
        )
        for r in beta:
            print(
                f"  {r['kind']:8s} n={r['n']} seed={r['seed']} Delta_beta={r['db']}: k=1 {r['local']:.2f}, collective {r['collective']:.2f} (h_max {r['h_max']:.2f}, mean weight {r['w_mean']:.2f})"
            )
            print("        P(400)       " + " ".join(f"{v:6.3f}" for v in r["P400"]))
            print(
                "        P(400)/P(50) "
                + " ".join(f"{x / max(y, 1e-12):6.2f}" for x, y in zip(r["P400"], r["P50"]))
            )
        cliffs = {}
        for r in beta:
            r["cliff"] = half_max_cliff(r["dg"], r["P400"])
            r["best400"], r["best50"] = (
                r["dg"][int(np.argmax(r["P400"]))],
                r["dg"][int(np.argmax(r["P50"]))],
            )
            cliffs.setdefault(r["db"], []).append(
                (
                    r["cliff"],
                    r["cliff"] / r["local"],
                    r["cliff"] / r["collective"],
                    r["best400"],
                    r["best50"],
                )
            )
        beta_summary = {}
        for db, v in cliffs.items():
            v = np.array(v)
            beta_summary[db] = dict(
                cliff=[v[:, 0].min(), float(np.median(v[:, 0])), v[:, 0].max()],
                over_single_flip=[v[:, 1].min(), v[:, 1].max()],
                over_collective=[v[:, 2].min(), v[:, 2].max()],
                best400=float(np.median(v[:, 3])),
                best50=float(np.median(v[:, 4])),
            )
            b = beta_summary[db]
            print(
                f"  Delta_beta={db}: cliff (half of the largest P(400)) {b['cliff'][0]:.2f}-{b['cliff'][2]:.2f}, median {b['cliff'][1]:.2f};  / k=1 {b['over_single_flip'][0]:.2f}-{b['over_single_flip'][1]:.2f};"
                f"  / collective {b['over_collective'][0]:.2f}-{b['over_collective'][1]:.2f};  best Delta_gamma at 400 / 50 layers (median) {b['best400']:.2f} / {b['best50']:.2f}"
            )
        dehn = dehn_comparison()
        print("dehn. published fit against the collective threshold with mean weight 0.5")
        for r in dehn:
            print(
                f"  N={r['N']:3d}{' (printed)' if r.get('printed') else '          '}: Delta_gamma {r['dg']:.3f}, Delta_beta {r['db']:.3f}, product x N {r['product_times_N']:.2f}, threshold {r['threshold']:.3f}, ratio {r['ratio']:.2f}"
            )
        bdest = [f.result() for f in f_bdest]
        print(
            "beta-dest. uniform weights, n = 14, 400 layers, past the cliff: where the probability goes"
        )
        for r in bdest:
            print(
                f"  seed={r['seed']} Delta_beta={r['db']} Delta_gamma={r['dg']} ({r['dg'] / r['local']:.2f} x k=1): P(x*)={r['P_opt']:.3f}, energy vs random {r['energy_vs_random']:.2f}, by Hamming distance 1..7: "
                + " ".join(f"{r['by_hamming'][d]:.3f}" for d in range(1, 8))
            )
            for t in r["top"][:2]:
                print(
                    f"        Hamming {t['hamming']}, energy +{t['dE']:.3f}: prob {t['prob']:.3f}   Delta_gamma*dE/2pi = {t['wraps']:.3f}"
                )

        fields = [f.result() for f in f_field]
        field = {}
        print(
            "field. largest local field in a best cut, weighted complete graphs, median of 5 seeds"
        )
        for signed, name in ((False, "positive weights"), (True, "signed weights")):
            ns = sorted({n for n, sg, _ in fields if sg == signed})
            med = [
                float(np.median([h for n, sg, h in fields if sg == signed and n == m])) for m in ns
            ]
            field[name] = dict(
                n=ns, h_max=med, exponent=float(np.polyfit(np.log(ns), np.log(med), 1)[0])
            )
            print(
                f"  {name:17s} "
                + " ".join(f"n={m}: {h:.2f}" for m, h in zip(ns, med))
                + f"   grows as n^{field[name]['exponent']:.2f}"
            )

    R = json.loads(OUT.read_text()) if OUT.exists() else {}
    for stale in (
        "q6_complete_graph_product_law",
        "q6_complete_graph",
    ):  # replaced by the stability table
        R.pop(stale, None)
    R.update(
        q6_complete_graph_stability=kn,
        q6_threshold_deep=deep,
        q6_deep_failure_destination=dest,
        q5_weighted_complete_beta_dependence=beta,
        q5_weighted_complete_destination=bdest,
        q6_bipartite_large_mixer=bip,
        q5_local_field_scaling=field,
        q5_weighted_complete_cliffs={str(k): v for k, v in beta_summary.items()},
        q5_dehn_comparison=dehn,
    )
    OUT.write_text(json.dumps(R, indent=1))
    print("wrote", OUT)
