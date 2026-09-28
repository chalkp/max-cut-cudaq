"""Checks on the ring and large-spin results.

  R  ring formula vs the sweep, exact vs fitted eta(p), depth for P = 1/2
  A  intercept: log2 P(n) + eta n - 1 -> 0, exponentially in n
  B  Landau-Zener: eta(p) sqrt(p) -> K(dg, db)
  C  eta as max(dg, db) crosses pi/2
  D  power-law ramps through the critical point, eta ~ p^(-r/(r+1))
  E  K(n/2, n/2) as two spins vs the sweep, gap position vs n
  F  both reductions vs the brute-force runs of gap_pilot.py
  G  Delta that minimizes K on the ring
  H  layers for P(x*) >= 1/2 on the ring, several schedules
  I  the published fidelity law is off by a factor 2

R, E and F read results/lr_p_sweep.csv (+ _eta.csv) and results/gap_pilot.csv,
so run those first.

    python experiments/solvable_models.py      # -> results/verification_numbers.json
"""

import json
import pathlib

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import zeta

from maxcut_cudaq.exact import (
    adiabatic_schedule,
    bipartite_success,
    eta_infinite,
    first_depth,
    lr_schedule,
    mode_probabilities,
    momenta,
    power_ramp,
    ring_log2_success,
    ring_success,
    spin_ops,
)

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "verification_numbers.json"
R = {}


def eta_slope(p, dg, db, N=1 << 14, u=None):
    return (ring_log2_success(N, p, dg, db, u=u) - ring_log2_success(2 * N, p, dg, db, u=u)) / N


# ------------------------------------------------------------------ A: the intercept
def check_intercept():
    print("A. intercept")
    ident = {}
    for n in (8, 14, 100, 1000):
        k = (2 * np.arange(1, n + 1) - 1) * np.pi / n
        ident[n] = float(np.sum(np.log2(2 - 2 * np.cos(k))))
    print(
        "   sum_j log2(2 - 2cos k_j) over the full antiperiodic grid:",
        {n: round(v, 12) for n, v in ident.items()},
    )
    rows = []
    for p in (5, 20, 100, 300):
        eta = eta_slope(p, 0.6, 0.3, N=1 << 15)
        k = (np.arange(4000) + 0.5) * np.pi / 4000
        ratio = mode_probabilities(k, *lr_schedule(p)) / (4 * np.sin(k / 2) ** 2)
        dev = {n: ring_log2_success(n, p) + eta * n - 1 for n in (8, 16, 32, 64, 128, 256, 512)}
        ns = [n for n, d in dev.items() if abs(d) > 1e-9]
        decay = (
            float(-np.polyfit(ns, np.log(np.abs([dev[n] for n in ns])), 1)[0])
            if len(ns) >= 3
            else float("nan")
        )
        rows.append(
            dict(
                p=p,
                eta=eta,
                min_ratio=float(ratio.min()),
                deviation={int(n): float(d) for n, d in dev.items()},
                decay_rate=decay,
                decay_length=1 / decay if decay == decay else None,
            )
        )
        print(
            f"   p={p:4d}: eta={eta:.6f}  min P_k/(4 sin^2(k/2)) = {ratio.min():.3e}  "
            f"log2P + eta n - 1: "
            + " ".join(f"n={n}:{d:+.1e}" for n, d in dev.items())
            + f"  decay length ~ {1 / decay:.1f}"
        )
    R["intercept"] = dict(identity=ident, rows=rows)


# ------------------------------------------------------------------ B: Landau-Zener
def K_pred(dg, db):
    gs = dg * db / (dg + db)  # gamma at the critical point
    c = np.pi * np.sin(2 * gs) ** 2 / (2 * (dg + db))  # P_exc(k) = exp(-c p k^2)
    return float(zeta(1.5) / (4 * np.sqrt(np.pi) * np.log(2) * np.sqrt(c))), float(c)


def check_landau_zener():
    print("B. Landau-Zener constant  eta sqrt(p) -> K")
    rows = []
    for dg, db in ((0.6, 0.3), (0.3, 0.3), (1.0, 1.0), (1.2, 0.5), (0.2, 0.9)):
        K, c = K_pred(dg, db)
        Kc = float(
            zeta(1.5)
            / (
                4
                * np.sqrt(np.pi)
                * np.log(2)
                * np.sqrt(np.pi * (2 * dg * db / (dg + db)) ** 2 / (2 * (dg + db)))
            )
        )
        num = {p: eta_slope(p, dg, db) * np.sqrt(p) for p in (1000, 10000)}
        rows.append(
            dict(
                dg=dg,
                db=db,
                K_floquet=K,
                K_continuum=Kc,
                numeric={int(p): float(v) for p, v in num.items()},
            )
        )
        print(
            f"   dg={dg}, db={db}: K predicted {K:.4f} (continuum coupling would give {Kc:.4f});  "
            + "  ".join(f"p={p}: {v:.4f}" for p, v in num.items())
        )
    v = eta_slope(100000, 0.6, 0.3) * np.sqrt(100000)
    print(f"   dg=0.6, db=0.3 at p=100000: {v:.5f}  (predicted {K_pred(0.6, 0.3)[0]:.5f})")
    R["landau_zener"] = dict(
        rows=rows, p1e5=float(v), half_depth_coefficient=K_pred(0.6, 0.3)[0] ** 2 / 4
    )
    print(f"   => layers for P = 1/2:  p = (K/2)^2 n^2 = {K_pred(0.6, 0.3)[0] ** 2 / 4:.4f} n^2")


# ------------------------------------------------------------------ C: pi/2 resonance
def check_resonance():
    print("C. eta across max(dg, db) = pi/2 = 1.5708")
    rows = []
    for label, pairs in (
        ("dg sweep, db=0.3", [(x, 0.3) for x in (1.2, 1.4, 1.5, 1.55, 1.6, 1.7, 2.0)]),
        ("db sweep, dg=0.6", [(0.6, x) for x in (1.2, 1.4, 1.5, 1.55, 1.6, 1.7, 2.0)]),
    ):
        print("   " + label)
        for dg, db in pairs:
            e = {p: eta_slope(p, dg, db, N=1 << 12) for p in (50, 200, 1000)}
            rows.append(dict(dg=dg, db=db, eta={int(p): float(v) for p, v in e.items()}))
            print(
                f"     dg={dg:<4} db={db:<4}: "
                + "  ".join(f"eta({p})={v:.4f}" for p, v in e.items())
            )
    R["resonance"] = rows


# ------------------------------------------------------------------ D: power-law ramps
def check_power_ramps():
    print("D. power-law passage through the critical point:  eta ~ p^(-r/(r+1))")
    ustar = 0.3 / 0.9
    rows = []
    for r in (1, 2, 3, 4):
        ps = [300, 1000, 3000, 10000]
        e = [eta_slope(p, 0.6, 0.3, u=power_ramp(p, r, ustar)) for p in ps]
        a = float(np.polyfit(np.log(ps[1:]), np.log(e[1:]), 1)[0])
        rows.append(
            dict(r=r, p=ps, eta=[float(x) for x in e], fitted_exponent=a, predicted=-r / (r + 1))
        )
        print(
            f"   r={r}: eta = "
            + ", ".join(f"{x:.5f}" for x in e)
            + f"   exponent {a:+.3f}  (predicted {-r / (r + 1):+.3f})"
        )
    R["power_ramps"] = rows


# ------------------------------------------------------------------ E: complete bipartite
def bipartite_gap(n, dg=0.6, db=0.3, grid=None):
    """(gap, u) of the smallest gap the ground state couples to, two-spin model."""
    m = n // 2
    M, Sx = spin_ops(m)
    I = sp.identity(m + 1)
    HM = -2 * (sp.kron(sp.csr_matrix(Sx), I) + sp.kron(I, sp.csr_matrix(Sx)))
    HC = sp.diags(4 * np.outer(M, M).ravel())
    V = -db * HM + dg * HC
    mf = db / (db + m * dg)
    grid = np.linspace(0.3 * mf, min(0.95, 8 * mf), 240) if grid is None else grid
    best = (np.inf, None)
    v0 = None
    for uu in grid:
        H = ((1 - uu) * db * HM + uu * dg * HC).tocsc()
        E, W = spla.eigsh(H, k=8, which="SA", v0=v0, tol=1e-10, maxiter=50000)
        o = np.argsort(E)
        E, W = E[o], W[:, o]
        v0 = W[:, 0]
        c = np.abs(W[:, 1:].T @ (V @ W[:, 0]))
        ok = (c > 1e-7) & (E[1:] - E[0] > 1e-10)
        if ok.any() and (E[1:] - E[0])[ok].min() < best[0]:
            best = (float((E[1:] - E[0])[ok].min()), float(uu))
    return best


def check_bipartite():
    print("E. complete bipartite graph as two large spins")
    d = pd.read_csv(RESULTS / "lr_p_sweep.csv")
    d = d[d.family == "bipartite"]
    g = d.groupby(["n_nodes", "p"]).agg(measured=("success_probability", "mean")).reset_index()
    g["exact"] = [bipartite_success(int(n), int(p)) for n, p in zip(g.n_nodes, g.p)]
    rel = np.abs(g.measured - g.exact) / g.exact
    print(
        f"   vs the CUDA-Q sweep: {len(g)} cells, max |measured - exact| = {np.abs(g.measured - g.exact).max():.2e}, "
        f"max relative deviation {rel.max():.2e}"
    )
    R["bipartite_validation"] = dict(
        cells=int(len(g)),
        max_abs_dev=float(np.abs(g.measured - g.exact).max()),
        max_rel_dev=float(rel.max()),
    )
    rows = []
    for n in (8, 16, 32, 64, 128):
        gap, ustar = bipartite_gap(n)
        mf = 0.3 / (0.3 + (n / 2) * 0.6)
        rows.append(dict(n=n, u_star=ustar, gap=gap, mean_field=mf))
        print(
            f"   n={n:4d}: gap minimum at u = {ustar:.4f} (mean field db/(db + n dg/2) = {mf:.4f}), "
            f"u*(n+1) = {ustar * (n + 1):.3f}, gap = {gap:.4f}"
        )
    a = np.polyfit(np.log([r["n"] for r in rows[1:]]), np.log([r["u_star"] for r in rows[1:]]), 1)[
        0
    ]
    print(f"   fitted u* ~ n^{a:+.3f}")
    R["bipartite_gap"] = dict(rows=rows, exponent=float(a))
    print("   best delta_gamma at p=50, db=0.3 (grid of 60 values in [0.01, 1.2]):")
    best = []
    for n in (16, 32, 64, 128):
        dgs = np.geomspace(0.01, 1.2, 60)
        ps = [bipartite_success(n, 50, dg, 0.3) for dg in dgs]
        i = int(np.argmax(ps))
        fixed = bipartite_success(n, 50, 0.6, 0.3)
        best.append(
            dict(
                n=n,
                best_dg=float(dgs[i]),
                best_P=float(ps[i]),
                P_at_0p6=float(fixed),
                dg_times_n=float(dgs[i] * n),
            )
        )
        print(
            f"     n={n:4d}: best dg = {dgs[i]:.4f} (dg*n = {dgs[i] * n:.2f}), P = {ps[i]:.4f};  at dg=0.6: P = {fixed:.2e}"
        )
    R["bipartite_delta"] = best


# ------------------------------------------------------------------ F: reductions vs brute force
def check_brute_force():
    print("F. reductions vs brute-force state vector (results/gap_pilot.csv, linear arm)")
    d = pd.read_csv(RESULTS / "gap_pilot.csv")
    d = d[(d.schedule == "linear") & (d.seed == 0)]
    out = {}
    for fam, fn in (
        ("ring", lambda n, p: 2.0 ** ring_log2_success(n, p)),
        ("bipartite", bipartite_success),
    ):
        g = d[d.family == fam]
        err = max(
            abs(fn(int(n), int(p)) - v) for n, p, v in zip(g.n_nodes, g.p, g.success_probability)
        )
        out[fam] = dict(cells=int(len(g)), max_abs_error=float(err))
        print(
            f"   {fam:9s}: {len(g)} cells (n = {sorted(g.n_nodes.unique())}, p = {sorted(g.p.unique())}), "
            f"max |reduction - brute force| = {err:.2e}"
        )
    R["brute_force"] = out


# ------------------------------------------------------------------ G: best Delta on the ring
def check_best_delta():
    print("G. Delta that minimises K on the ring, inside the box max(dg, db) < pi/2")
    grid = np.linspace(0.05, np.pi / 2 - 1e-3, 400)
    DG, DB = np.meshgrid(grid, grid, indexing="ij")
    Kmap = np.sqrt(DG + DB) / np.sin(2 * DG * DB / (DG + DB))
    i, j = np.unravel_index(np.argmin(Kmap), Kmap.shape)
    dg, db = float(grid[i]), float(grid[j])
    K, _ = K_pred(dg, db)
    num = eta_slope(10000, dg, db) * 100
    print(
        f"   minimum at dg = {dg:.4f}, db = {db:.4f}: K = {K:.4f} (numeric eta sqrt(p) at p=1e4: {num:.4f}); "
        f"paper default (0.6, 0.3): K = {K_pred(0.6, 0.3)[0]:.4f}; (1, 1): K = {K_pred(1.0, 1.0)[0]:.4f}"
    )
    print(
        f"   layers for P = 1/2: {K**2 / 4:.4f} n^2 at the optimum vs {K_pred(0.6, 0.3)[0] ** 2 / 4:.4f} n^2 at the default"
    )
    R["best_delta"] = dict(
        dg=dg,
        db=db,
        K=K,
        numeric=float(num),
        K_default=K_pred(0.6, 0.3)[0],
        K_one_one=K_pred(1.0, 1.0)[0],
    )


# ------------------------------------------------------------------ H: how far below n^2 can a schedule go
def gap_power_schedule(n, p, q, dg=0.6, db=0.3, grid=400001):
    """du/dt ~ gap(u)^q, gap = 2 |B_k1(u)| of the slowest mode k1 = pi/n.

    q = 2 is Roland-Cerf, Guo and An single out q = 3/2.
    """
    k1 = np.pi / n
    u = np.linspace(0, 1, grid)
    bz, bx = -2 * db * (1 - u) + 2 * dg * u * np.cos(k1), -2 * dg * u * np.sin(k1)
    rate = (2 * np.sqrt(bz**2 + bx**2)) ** (-q)
    tau = np.concatenate([[0], np.cumsum((rate[1:] + rate[:-1]) / 2)])
    return np.interp(np.linspace(0, 1, p + 1), tau / tau[-1], u)


def check_schedule_depths():
    print(
        "H. layers for P(x*) >= 1/2 on the ring (5%-spaced search grid); variational optimum is n/2"
    )
    sizes = [32, 64, 128, 256, 512]
    table = {}
    # exact coefficient of n^2 for the linear ramp (bisection, monotone in p)
    coef = {}
    for n in (16, 32, 64, 128, 256):
        lo, hi = 1, 4 * n * n
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if ring_log2_success(n, mid) >= -1 else (mid, hi)
        coef[n] = hi / n**2
    print(
        "   linear ramp, exact p_half/n^2:",
        {n: round(v, 4) for n, v in coef.items()},
        " K^2/4 =",
        round(K_pred(0.6, 0.3)[0] ** 2 / 4, 4),
    )
    arms = [("pilot A(u)^-1/2", lambda n: lambda p: adiabatic_schedule(n, p))]
    arms += [
        (f"gap^{q}", (lambda q: lambda n: lambda p: gap_power_schedule(n, p, q))(q))
        for q in (1.0, 1.25, 1.5, 1.75, 2.0)
    ]
    ustar = 0.3 / 0.9
    arms += [
        (f"power r={r}", (lambda r: lambda n: lambda p: power_ramp(p, r, ustar))(r))
        for r in (2, 4, 8, 16)
    ]
    for name, mk in arms:
        ps = [first_depth(n, mk(n)) for n in sizes]
        a = float(np.polyfit(np.log(sizes[-3:]), np.log(ps[-3:]), 1)[0])
        table[name] = dict(
            n=sizes, p_half=ps, local_exponent=a, p_over_n=[p / n for p, n in zip(ps, sizes)]
        )
        print(
            f"   {name:16s}: "
            + ", ".join(f"{p:5d}" for p in ps)
            + f"   local exponent {a:.2f}   p/n at n=512: {ps[-1] / 512:.2f}"
        )
    R["schedule_depths"] = dict(linear_coefficient=coef, table=table)
    # the n = 8, p = 4 ring data point of arXiv:2606.28536 (LR-QAOA with both Deltas optimised): 0.387
    g = np.linspace(0.05, 1.57, 153)
    best = max((2.0 ** ring_log2_success(8, 4, dg, db), dg, db) for dg in g for db in g)
    print(
        f"   n=8, p=4, best P(x*) over a 153 x 153 grid of (dg, db) below pi/2: {best[0]:.4f} at dg={best[1]:.2f}, db={best[2]:.2f}  (arXiv:2606.28536 reports 0.387)"
    )
    R["n8_p4_best"] = dict(P=float(best[0]), dg=float(best[1]), db=float(best[2]))


# ------------------------------------------------------------------ I: the published fidelity law and its missing factor
def check_published_law():
    """Cincio et al. 2009 (Eq. 41) and Dziarmaga 2010 (Eq. 158) get F = exp(-1.3 d N)
    for the continuous linear quench by turning the mode sum into an integral. The
    exact product is twice that.
    """
    print("I. published law F = exp(-(zeta(3/2)/2) d N) against the exact product")
    c = float(zeta(1.5) / 2)
    rows = []
    for tau, N in ((2, 200), (10, 400), (10, 2000), (50, 2000), (200, 20000)):
        k = (2 * np.arange(1, N // 2 + 1) - 1) * np.pi / N
        lnF = float(np.sum(np.log1p(-np.exp(-2 * np.pi * tau * k * k))))
        dN = N / (2 * np.pi * np.sqrt(2 * tau))
        rows.append(dict(tau=tau, N=N, kinks=dN, lnF_exact=lnF, ratio=float(np.exp(lnF + c * dN))))
        print(
            f"   continuous quench tau={tau:4d} N={N:6d}: exact / published = {rows[-1]['ratio']:.6f}"
        )
    circ = []
    for n, p in ((1024, 50), (4096, 300), (4096, 1000), (16384, 3000)):
        Pk = mode_probabilities(momenta(n), *lr_schedule(p))
        kinks = float(2 * np.sum(1 - Pk))
        lnP = float(np.sum(np.log(Pk)))
        circ.append(
            dict(n=n, p=p, kinks=kinks, with_ln2=(-lnP + np.log(2)) / kinks, without=-lnP / kinks)
        )
        print(
            f"   LR-QAOA circuit n={n:6d} p={p:5d}: (-ln P + ln 2)/<kinks> = {circ[-1]['with_ln2']:.4f}, "
            f"without ln 2: {circ[-1]['without']:.4f}  (zeta(3/2)/2 = {c:.4f})"
        )
    R["published_law"] = dict(coefficient=c, continuous=rows, circuit=circ)


# ------------------------------------------------------------------ R: ring formula vs the sweep, eta tables
def check_ring():
    print("R. ring formula against the LR p-sweep; eta(p); depth for P = 1/2; eta over (dg, db)")
    d = pd.read_csv(RESULTS / "lr_p_sweep.csv")
    d = d[d.family == "ring"]
    g = d.groupby(["n_nodes", "p"]).agg(measured=("success_probability", "mean")).reset_index()
    g["exact"] = [ring_success(int(n), int(p)) for n, p in zip(g.n_nodes, g.p)]
    g["rel_dev"] = np.abs(g.measured - g.exact) / g.exact
    print(
        f"ring cells (n, p): {len(g)};  max |measured - exact| = {np.abs(g.measured - g.exact).max():.2e};  "
        f"max relative deviation {g.rel_dev.max():.2e}"
    )
    R["ring_validation"] = dict(
        cells=int(len(g)),
        max_abs_dev=float(np.abs(g.measured - g.exact).max()),
        max_rel_dev=float(g.rel_dev.max()),
    )
    print(
        g[g.n_nodes.isin([8, 16, 28]) & g.p.isin([5, 20, 100, 300])]
        .round(4)
        .to_string(index=False),
        "\n",
    )

    fit = pd.read_csv(RESULTS / "lr_p_sweep_eta.csv")
    fit = fit[fit.family == "ring"].set_index("p")
    rows = []
    for p in sorted(d.p.unique()):
        e = eta_infinite(int(p))
        C = ring_log2_success(1 << 14, int(p)) + e * (1 << 14)
        rows.append(
            dict(
                p=int(p),
                eta_exact=e,
                C_exact=C,
                eta_fit_n8_28=fit.loc[p, "eta"],
                C_fit=fit.loc[p, "C"],
            )
        )
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    for lo, hi in ((5, 60), (60, 300), (300, 3000)):
        ps = np.array(
            [p for p in (5, 10, 20, 30, 60, 100, 200, 300, 600, 1000, 3000) if lo <= p <= hi]
        )
        es = np.array([eta_infinite(int(p)) for p in ps])
        print(f"eta ~ p^a on p in [{lo},{hi}]: a = {np.polyfit(np.log(ps), np.log(es), 1)[0]:+.3f}")

    def depth_for_half(n, adaptive):
        p = 2
        while ring_log2_success(n, p, u=adiabatic_schedule(n, p) if adaptive else None) < -1.0:
            p = max(p + 1, int(p * 1.05))
        return p

    print(
        "\ndepth needed for P(x*) >= 0.5 (first p on a 5%-spaced grid): linear ramp vs gap-adapted"
    )
    rows = []
    for n in (16, 32, 64, 100, 128, 256):
        pl, pa = depth_for_half(n, False), depth_for_half(n, True)
        rows.append(
            dict(
                n=n,
                p_linear=pl,
                p_adapted=pa,
                saving=pl / pa,
                lin_over_n2=pl / n**2,
                adapted_over_n=pa / n,
            )
        )
    t = pd.DataFrame(rows)
    print(t.round(3).to_string(index=False))
    print(
        "scaling exponents p ~ n^a:  linear a = %.2f,  adapted a = %.2f"
        % (
            np.polyfit(np.log(t.n), np.log(t.p_linear), 1)[0],
            np.polyfit(np.log(t.n), np.log(t.p_adapted), 1)[0],
        )
    )

    print("\nat hardware-reachable depth on n = 100 (P(x*), linear vs gap-adapted):")
    for p in (5, 10, 15, 50, 100):
        print(
            f"  p={p:4d}: linear 2^{ring_log2_success(100, p):8.2f}   adapted 2^{ring_log2_success(100, p, u=adiabatic_schedule(100, p)):8.2f}"
        )

    print("\neta(p=50) over the (delta_gamma, delta_beta) plane (rows dg, cols db):")
    dgs, dbs = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4), (0.1, 0.2, 0.3, 0.4, 0.6, 0.8)
    print(
        pd.DataFrame(
            [[eta_infinite(50, dg, db, N=1 << 11) for db in dbs] for dg in dgs],
            index=dgs,
            columns=dbs,
        )
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    check_ring()
    check_intercept()
    check_landau_zener()
    check_resonance()
    check_power_ramps()
    check_bipartite()
    check_brute_force()
    check_best_delta()
    check_schedule_depths()
    check_published_law()
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(R, indent=1))
    print("wrote", OUT)
