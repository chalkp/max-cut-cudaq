"""Cross-check against upstream max-cut-cudaq on random cases.

  unitary  n = 4..10, the full circuit matrix: Qiskit Operator vs cudaq.get_unitary
  state    n = 11..20, the state vector, P(x*) and expected cut: Aer vs CUDA-Q
  sweep    rows of upstream's lr_p_sweep.csv (dense up to n = 28), rerun here

--hardware reruns the cells of a hardware.py run on CUDA-Q instead and sets
them against what the device (or its model) measured.

CUDA-Q builds e^{-itZZ} from r1/cr1 and drops e^{-it}, so its matrices carry a
global phase e^{+i sum_k gamma_k sum wn}, taken out before comparing. Its
get_unitary puts qubit 0 in the top bit, its states in the bottom bit.

The CUDA-Q side needs its own environment, so this script calls itself there
with --dump and reads the arrays back. Upstream is ../max-cut-cudaq unless
--cudaq says otherwise.

    compare_cudaq.py
    compare_cudaq.py --hardware results/hardware/sweep_fake_kawasaki_ideal.json
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "max-cut-cudaq"
FAMILIES = ["ring", "ramanujan", "clustered", "bipartite", "erdos_renyi", "euclidean", "dense"]
WEIGHTS = np.array([1, 1, 1, 1, 1, 1, 2]) / 8  # dense twice as often


def bit_reverse(n):
    x = np.arange(1 << n)
    r = np.zeros_like(x)
    for q in range(n):
        r |= ((x >> q) & 1) << (n - 1 - q)
    return r


def dump(cases_path, out_path):
    # runs in the CUDA-Q environment
    import cudaq
    from maxcut_cudaq import engine
    from maxcut_cudaq.qaoa import Problem

    out = {}
    for i, c in enumerate(json.loads(pathlib.Path(cases_path).read_text())):
        P = Problem(c["src"], c["tgt"], c["w"], n=c["n"], backend="cudaq")
        g, b = c["gammas"], c["betas"]
        t0 = time.time()
        if c["kind"] == "unitary":
            engine.select_target(P.n, "fp64")
            args = (P.src.tolist(), P.tgt.tolist(), P.wn.tolist(), P.deg.tolist(), g, b)
            U = np.array(cudaq.get_unitary(engine.qaoa_circuit, P.n, *args))
            r = bit_reverse(P.n)
            out[f"{i}"] = U[np.ix_(r, r)]
        elif c["kind"] == "state":
            _, amp = engine.state(P, g, b, "fp64")
            out[f"{i}"] = amp.get()
            res = P.evaluate(g, b)
            out[f"{i}_r"] = np.array([res["success_probability"], res["expected_cut"]])
        else:
            res = P.evaluate(g, b)
            out[f"{i}_r"] = np.array([res["success_probability"], res["mean_ratio"]])
        out[f"{i}_t"] = np.array(time.time() - t0)
    np.savez(out_path, **out)


def draw_cases(rng, kind, count, n_lo, n_hi, p_hi):
    """Random instances; half get a linear ramp with random Delta, half random angles."""
    from maxcut_qiskit.graphs import edge_arrays, make_graph

    cases = []
    while len(cases) < count:
        family = str(rng.choice(FAMILIES, p=WEIGHTS))
        n, seed = int(rng.integers(n_lo, n_hi + 1)), int(rng.integers(1000))
        try:
            G = make_graph(family, n, seed)
        except (ValueError, RuntimeError):
            continue  # odd n for ramanujan, n < 8 for clustered
        p = int(rng.integers(1, p_hi + 1))
        if rng.random() < 0.5:
            dg, db = rng.uniform(0.1, 1.5), rng.uniform(0.1, 1.0)
            u = np.linspace(0, 1, p + 1)
            g, b, schedule = u[1:] * dg, (1 - u[:-1]) * db, f"lr({dg:.2f},{db:.2f})"
        else:
            g, b, schedule = rng.uniform(-np.pi, np.pi, p), rng.uniform(-np.pi, np.pi, p), "random"
        src, tgt, w = edge_arrays(G)
        cases.append(
            dict(
                kind=kind,
                family=family,
                n=n,
                seed=seed,
                p=p,
                schedule=schedule,
                src=src.tolist(),
                tgt=tgt.tolist(),
                w=w.tolist(),
                gammas=g.tolist(),
                betas=b.tolist(),
            )
        )
    return cases


def cudaq_python(path):
    if shutil.which("pixi"):
        return ["pixi", "run", "--manifest-path", str(path / "pyproject.toml"), "python"]
    return [str(path / ".venv" / "bin" / "python")]


def cudaq_reference(cases, upstream):
    """Run @cases in the CUDA-Q environment; arrays keyed by case index."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PIXI_", "CONDA_"))}
    with tempfile.TemporaryDirectory() as tmp:
        cp, npz = pathlib.Path(tmp, "cases.json"), pathlib.Path(tmp, "cudaq.npz")
        cp.write_text(json.dumps(cases))
        me = str(pathlib.Path(__file__).resolve())
        cmd = [*cudaq_python(upstream), me, "--dump", str(cp), str(npz)]
        subprocess.run(cmd, check=True, env=env, cwd=upstream)
        return dict(np.load(npz))


def phase_error(a, b, theta):
    """max |a - e^{-i theta} b|, and how far the best-fit phase is from -theta."""
    fit = np.angle(np.vdot(b, a))
    err = float(np.abs(a - np.exp(-1j * theta) * b).max())
    return err, float(abs(np.angle(np.exp(1j * (fit + theta)))))


def compare_circuits(cases, upstream):
    from qiskit.quantum_info import Operator

    from maxcut_qiskit import aer
    from maxcut_qiskit.qaoa import Problem

    ref = cudaq_reference(cases, upstream)
    rows = []
    for i, c in enumerate(cases):
        P = Problem(c["src"], c["tgt"], c["w"], n=c["n"], backend="aer")
        g, b = np.array(c["gammas"]), np.array(c["betas"])
        theta = float(g.sum() * P.wn.sum())
        qc = aer.circuit(P, g, b)
        t0 = time.time()
        if c["kind"] == "unitary":
            mine = Operator(qc).data
        else:
            qc.save_statevector(label="psi")
            mine = np.asarray(aer.run(qc, "fp64")["psi"])
            res = P.evaluate(g, b)
        dt = time.time() - t0
        err, dphase = phase_error(mine.ravel(), ref[f"{i}"].ravel(), theta)
        row = {k: c[k] for k in ("kind", "family", "n", "seed", "p", "schedule")}
        row.update(max_abs_err=err, phase_err=dphase, qiskit_s=dt, cudaq_s=float(ref[f"{i}_t"]))
        if c["kind"] == "state":
            ps, ec = ref[f"{i}_r"]
            row.update(
                pstar_rel_err=abs(res["success_probability"] - ps) / ps,
                cut_rel_err=abs(res["expected_cut"] - ec) / abs(ec),
            )
        rows.append(row)
    return rows


def compare_sweep(rng, upstream, count):
    import pandas as pd

    from maxcut_qiskit.graphs import edge_arrays, make_graph
    from maxcut_qiskit.qaoa import Problem, set_precision

    csv = upstream / "results" / "lr_p_sweep.csv"
    if not csv.exists():
        print(f"no {csv}, skipping the sweep rows (run upstream's sweep first)")
        return []
    d = pd.read_csv(csv)
    big = d[d.n_nodes >= 14]
    dense = big[big.family == "dense"]
    k28 = max(1, count // 10)
    pick = pd.concat(
        [
            dense[dense.n_nodes == 28].sample(k28, random_state=rng.integers(2**31)),
            dense[dense.n_nodes < 28].sample(
                max(0, count // 3 - k28), random_state=rng.integers(2**31)
            ),
            big[big.family != "dense"].sample(count - count // 3, random_state=rng.integers(2**31)),
        ]
    )
    rows = []
    for r in pick.sort_values(["n_nodes", "family"]).itertuples():
        set_precision(r.precision)
        P = Problem(*edge_arrays(make_graph(r.family, r.n_nodes, r.seed)), n=r.n_nodes)
        t0 = time.time()
        res = P.lr(r.p, r.delta_gamma, r.delta_beta)
        rows.append(
            dict(
                kind="sweep",
                family=r.family,
                n=r.n_nodes,
                seed=r.seed,
                p=r.p,
                precision=r.precision,
                optimum_equal=P.optimum() == r.ground_truth,
                pstar_rel_err=abs(res["success_probability"] - r.success_probability)
                / r.success_probability,
                ratio_abs_err=abs(res["mean_ratio"] - r.mean_ratio),
                qiskit_s=time.time() - t0,
                cudaq_s=r.wall_clock_s,
            )
        )
        print(
            f"  sweep n={r.n_nodes} {r.family} seed={r.seed} p={r.p} [{r.precision}]: "
            f"dP/P {rows[-1]['pstar_rel_err']:.1e}  "
            f"{rows[-1]['qiskit_s']:.1f}s vs {r.wall_clock_s:.1f}s",
            flush=True,
        )
    return rows


def compare_hardware(manifest, upstream):
    """compare_hardware() - a hardware.py run against CUDA-Q
    @manifest: the results/hardware/*.json written by hardware.py
    @upstream: the CUDA-Q project

    Every cell is rerun on the CUDA-Q state vector with the same angles. z is
    shot noise only, so ~N(0, 1) on a noiseless model and a noise signal on
    a device.
    """
    import pandas as pd

    from maxcut_qiskit.graphs import edge_arrays, make_graph

    m = json.loads(pathlib.Path(manifest).read_text())
    hw = pd.read_csv(pathlib.Path(manifest).with_suffix(".csv"))
    cases = []
    for c in m["cells"]:
        src, tgt, w = edge_arrays(make_graph(c["family"], c["n"], c["seed"]))
        cases.append(
            dict(
                kind="score",
                n=c["n"],
                src=src.tolist(),
                tgt=tgt.tolist(),
                w=w.tolist(),
                gammas=c["gammas"],
                betas=c["betas"],
            )
        )
    ref = cudaq_reference(cases, upstream)
    hw["P_cudaq"] = [float(ref[f"{i}_r"][0]) for i in range(len(cases))]
    hw["ratio_cudaq"] = [float(ref[f"{i}_r"][1]) for i in range(len(cases))]
    hw["P_err"] = np.sqrt(hw.success_probability * (1 - hw.success_probability) / hw.shots)
    hw["P_hw/P_cudaq"] = hw.success_probability / hw.P_cudaq
    hw["P_mit/P_cudaq"] = hw.success_probability_mitigated / hw.P_cudaq
    hw["z_P"] = (hw.success_probability - hw.P_cudaq) / np.sqrt(
        hw.P_cudaq * (1 - hw.P_cudaq) / hw.shots
    )
    hw["z_ratio"] = (hw.mean_ratio - hw.ratio_cudaq) / hw.mean_ratio_err
    hw["qiskit_vs_cudaq"] = (hw.success_probability_exact - hw.P_cudaq).abs() / hw.P_cudaq
    out = pathlib.Path(manifest).with_name(pathlib.Path(manifest).stem + "_vs_cudaq.csv")
    hw.to_csv(out, index=False)
    cols = ["family", "n", "p", "schedule", "router", "n_2q", "fidelity"]
    cols += ["success_probability", "P_err", "P_cudaq", "z_P", "P_hw/P_cudaq", "P_mit/P_cudaq"]
    cols += ["mean_ratio", "ratio_cudaq", "z_ratio"]
    print(hw[cols].to_string(index=False, float_format="%.4f"))
    print(
        f"\nz over {len(hw)} cells: P(x*) mean {hw.z_P.mean():+.2f} sd {hw.z_P.std():.2f}, "
        f"ratio mean {hw.z_ratio.mean():+.2f} sd {hw.z_ratio.std():.2f}"
    )
    print(f"Qiskit exact vs CUDA-Q: max rel diff {hw.qiskit_vs_cudaq.max():.1e}\nwrote {out}")
    return 0


def summary(rows):
    import pandas as pd

    d = pd.DataFrame(rows)
    errs = ["max_abs_err", "phase_err", "pstar_rel_err", "cut_rel_err", "ratio_abs_err"]
    g = d.groupby("kind")
    out = pd.concat(
        [
            g.size().rename("cases"),
            g.n.min().rename("n_min"),
            g.n.max().rename("n_max"),
            g[[c for c in errs if c in d]].max(),
            g[["qiskit_s", "cudaq_s"]].sum(),
        ],
        axis=1,
    )
    return out, d.groupby(["kind", "family"]).size().unstack(fill_value=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cudaq", default=str(UPSTREAM), help="the max-cut-cudaq checkout")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--unitary", type=int, default=60, help="random n = 4..10 unitaries")
    ap.add_argument("--state", type=int, default=30, help="random n = 11..20 state vectors")
    ap.add_argument("--sweep", type=int, default=30, help="upstream sweep rows to rerun")
    ap.add_argument("--out", default=str(ROOT / "results" / "compare_cudaq.json"))
    ap.add_argument("--hardware", help="compare a hardware.py manifest instead")
    ap.add_argument("--dump", nargs=2, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.dump:
        return dump(*a.dump)
    upstream = pathlib.Path(a.cudaq).resolve()
    if a.hardware:
        return compare_hardware(a.hardware, upstream)
    rng = np.random.default_rng(a.seed)
    cases = draw_cases(rng, "unitary", a.unitary, 4, 10, 8)
    cases += draw_cases(rng, "state", a.state, 11, 20, 60)
    rows = compare_circuits(cases, upstream)
    rows += compare_sweep(rng, upstream, a.sweep) if a.sweep else []
    out, fam = summary(rows)
    pathlib.Path(a.out).parent.mkdir(exist_ok=True)
    pathlib.Path(a.out).write_text(
        json.dumps(dict(seed=a.seed, rows=rows), indent=1, default=float)
    )
    print(out.to_string(float_format="%.2e"), "\n", fam.to_string(), f"\nwrote {a.out}", sep="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
