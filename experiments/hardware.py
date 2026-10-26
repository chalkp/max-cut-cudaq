"""LR-QAOA on UTokyo's IBM Quantum System One (ibm_kawasaki), scored exactly.

  sweep           LR ramp, Delta = (0.6, 0.3), P(x*) and cut ratio against p
  pilot           linear vs gap-adapted vs the two cubic schedules
  depth-matched   LR against COBYLA angles optimised on the exact state

Each cell (instance, angles) is one ISA circuit and one PUB. Cells whose odds of
running error-free (ibm.fidelity) fall below --min-fid are dropped; at 4000
shots they cannot be told from noise. Two readout calibration circuits ride
along, so every row has P(x*) and the cut ratio raw and readout-corrected.

    hardware.py sweep --n 8 10 --p 1 2 3 5              # fake_kawasaki, local
    hardware.py sweep --noiseless                       # same circuits, ideal
    hardware.py sweep --backend ibm_kawasaki --simulate # live calibration, local
    hardware.py sweep --backend ibm_kawasaki            # estimate only
    hardware.py sweep --backend ibm_kawasaki --submit   # job ids -> results/hardware
    hardware.py collect results/hardware/sweep_ibm_kawasaki.json
"""

import argparse
import csv
import datetime
import json
import pathlib
import sys
import time

import numpy as np

from maxcut_qiskit import ibm, schedules
from maxcut_qiskit.graphs import MIN_N, edge_arrays, make_graph
from maxcut_qiskit.qaoa import DB, DG, Problem, angles

OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "hardware"
FAMILIES = ["ring", "ramanujan", "clustered", "bipartite", "erdos_renyi", "euclidean", "dense"]
PILOT_FAMILIES = ["ring", "bipartite", "ramanujan", "irregular", "clustered", "erdos_renyi"]
PILOT_FAMILIES += ["euclidean", "heterogeneous_weight", "dense"]
DETERMINISTIC = {"ring", "bipartite"}

_problems = {}


def problem(family, n, seed):
    key = (family, n, seed)
    if key not in _problems:
        _problems[key] = Problem(*edge_arrays(make_graph(family, n, seed)), n=n)
    return _problems[key]


def instances(a, families):
    for n in a.n:
        for family in families:
            if n < MIN_N.get(family, 0):
                continue
            for seed in [0] if family in DETERMINISTIC else a.seeds:
                yield family, n, seed


def sweep_cells(a):
    for family, n, seed in instances(a, a.families or FAMILIES):
        for p in a.p:
            yield dict(family=family, n=n, seed=seed, p=p, schedule="linear"), angles(p, DG, DB)


def pilot_cells(a):
    for family, n, seed in instances(a, a.families or PILOT_FAMILIES):
        A = schedules.leak_weight(problem(family, n, seed))
        for p in a.p:
            for name in schedules.NAMES:
                u = schedules.schedule(name, p, A)
                yield dict(family=family, n=n, seed=seed, p=p, schedule=name), angles(p, u=u)


def depth_matched_cells(a):
    for family, n, seed in instances(a, a.families or FAMILIES):
        P = problem(family, n, seed)
        for p in a.p:
            cell = dict(family=family, n=n, seed=seed, p=p)
            yield dict(cell, schedule="cobyla"), schedules.cobyla(P, p, seed)
            yield dict(cell, schedule="linear"), angles(p, DG, DB)


# The grids ask the question and --min-fid keeps what the device can answer:
# Eagle keeps ~45% of the sweep and ~30% of the rest, mostly sparse graphs at
# p <= 10. The p = 3, 5 rows give dense graphs a chance.
STUDIES = {
    "sweep": (sweep_cells, [8, 10, 12, 14, 16], [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]),
    "pilot": (pilot_cells, [8, 10, 12], [3, 5, 10, 20]),
    "depth-matched": (depth_matched_cells, [8, 10, 12], [3, 5, 10, 20]),
}


def build(a, bk):
    """Cells, their ISA circuits and the exact references. PUBs are the cells
    in order, then the all-0 and all-1 calibrations of every qubit read."""
    make, *_ = STUDIES[a.study]
    target, lines, cells, isas = bk.target, {}, [], []
    t0, dropped = time.time(), 0
    for meta, (g, b) in make(a):
        P = problem(meta["family"], meta["n"], meta["seed"])
        if meta["n"] not in lines:
            lines[meta["n"]] = ibm.best_line(target, meta["n"])
        isa, router = ibm.transpile(P, g, b, target, line=lines[meta["n"]])
        n2q, fid = ibm.count_2q(isa), ibm.fidelity(isa, target)
        if (a.max_2q and n2q > a.max_2q) or fid < a.min_fid:
            dropped += 1
            continue
        ref = P.evaluate(g, b)
        meta.update(
            gammas=[float(x) for x in g],
            betas=[float(x) for x in b],
            router=router,
            n_2q=n2q,
            depth=isa.depth(),
            fidelity=fid,
            measured=list(ibm.measured_qubits(isa)),
            qpu_s=ibm.qpu_seconds(isa, target, a.shots),
            success_probability_exact=ref["success_probability"],
            mean_ratio_exact=ref["mean_ratio"],
            random_success_probability=len(P.optimal_indices()) / 2**P.n,
        )
        cells.append(meta)
        isas.append(isa)
        print(
            f"  {meta['family']:11s} n={meta['n']:2d} seed={meta['seed']} p={meta['p']:3d} "
            f"{meta['schedule']:11s} {router:5s} 2q={n2q:5d} depth={meta['depth']:5d} "
            f"F={fid:.3f} P*exact={ref['success_probability']:.4f}",
            flush=True,
        )
    cal = sorted({q for c in cells for q in c["measured"]})
    isas += ibm.calibration(cal, bk.num_qubits) if cells else []
    print(
        f"[build] {len(cells)} circuits + 2 readout calibrations ({len(cal)} qubits) in "
        f"{time.time() - t0:.0f} s; {dropped} dropped",
        flush=True,
    )
    return cells, cal, isas


def score_rows(manifest, pubs):
    cells, k = manifest["cells"], len(manifest["cells"])
    e01, e10 = ibm.flip_rates(pubs[k].data.meas, pubs[k + 1].data.meas)
    at = {q: i for i, q in enumerate(manifest["calibration"])}
    rows = []
    for c, pub in zip(cells, pubs):
        i = [at[q] for q in c["measured"]]
        row = {key: v for key, v in c.items() if key not in ("gammas", "betas", "measured")}
        row.update(
            ibm.score(problem(c["family"], c["n"], c["seed"]), pub.data.meas, (e01[i], e10[i]))
        )
        row.update(backend=manifest["backend"], created=manifest["created"])
        rows.append(row)
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def summarize(rows):
    import pandas as pd

    d = pd.DataFrame(rows)
    d["kept"] = d.success_probability / d.success_probability_exact
    d["kept_mit"] = d.success_probability_mitigated / d.success_probability_exact
    cols = ["success_probability", "success_probability_mitigated", "success_probability_exact"]
    keys = ["family", "n", "schedule"] if d.schedule.nunique() > 1 else ["family", "n", "p"]
    print(d.groupby(keys)[cols + ["fidelity", "kept", "kept_mit"]].mean().round(4).to_string())


def device_line(bk):
    two, ro = ibm._errors(bk.target)
    e2, e_ro = np.median(list(two.values())), np.median(list(ro.values()))
    gates = "/".join(g for g in ibm.TWO_Q if g in bk.target.operation_names)
    return (
        f"{bk.name}: {ibm.processor(bk)}, {bk.num_qubits} qubits, "
        f"2q {gates} median error {e2:.1e}, readout {e_ro:.3f}"
    )


def run(a):
    real = not a.backend.startswith("fake_")
    if a.simulate and not real:
        raise SystemExit("--simulate is for a real device; fake_* already runs locally")
    bk = ibm.backend(a.backend, fractional=not a.no_fractional, instance=a.instance)
    print(f"[{a.study}] {device_line(bk)}", flush=True)
    cells, cal, isas = build(a, bk)
    if not cells:
        raise SystemExit("no circuits left; lower --min-fid or pick smaller n, p")
    qpu = sum(ibm.qpu_seconds(isa, bk.target, a.shots) for isa in isas)
    print(f"[estimate] {len(isas)} PUBs x {a.shots} shots, ~{qpu / 60:.1f} QPU minutes")
    local = not real or a.simulate or a.noiseless
    suffix = "_ideal" if a.noiseless else "_sim" if a.simulate else ""
    path = OUT / f"{a.study.replace('-', '_')}_{bk.name}{suffix}.json"
    manifest = dict(
        study=a.study,
        backend=bk.name,
        processor=ibm.processor(bk),
        fractional="rzz" in bk.target.operation_names,
        where="local" if local else "device",
        noiseless=a.noiseless,
        shots=a.shots,
        dd=not a.no_dd,
        twirl=a.twirl,
        min_fid=a.min_fid,
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        cells=cells,
        calibration=cal,
        jobs=[],
    )
    if a.dry_run or not (local or a.submit):
        print("[dry run] nothing run or sent" + ("; --submit sends them" if real else ""))
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    if not local:
        from contextlib import nullcontext

        from qiskit_ibm_runtime import Batch

        chunks = [isas[i : i + a.per_job] for i in range(0, len(isas), a.per_job)]
        with Batch(backend=bk) if a.mode == "batch" else nullcontext(bk) as mode:
            s = ibm.sampler(mode, dd=not a.no_dd, twirl=a.twirl)
            for ch in chunks:
                job = s.run(ch, shots=a.shots)
                manifest["jobs"].append(job.job_id())
                path.write_text(json.dumps(manifest, indent=1))
                print(f"[submit] job {job.job_id()}: {len(ch)} PUBs", flush=True)
        print(f"[submit] {len(chunks)} jobs; collect with: python {sys.argv[0]} collect {path}")
        return 0
    # XXX: one noisy Aer run holding several ISA circuits can return zeros for
    # some of them (qiskit-aer 0.17-0.18, CPU). Real devices are fine, and the
    # method depends on the width anyway, so local PUBs go one at a time.
    sims, pubs, t0 = {}, [], time.time()
    for isa in isas:
        n = ibm.width(isa)
        key = ibm.sim_method(n, a.noiseless)
        if key not in sims:
            sims[key] = ibm.simulator(bk, n, a.noiseless)
        pubs.append(sims[key].run([isa], shots=a.shots).result()[0])
    print(f"[local] {len(pubs)} PUBs in {time.time() - t0:.0f} s", flush=True)
    manifest["jobs"].append("local")
    path.write_text(json.dumps(manifest, indent=1))
    rows = score_rows(manifest, pubs)
    write_csv(path.with_suffix(".csv"), rows)
    summarize(rows)
    return 0


def collect(a):
    path = pathlib.Path(a.manifest)
    manifest = json.loads(path.read_text())
    service = ibm.account(a.instance)
    pubs, job_of, usage = [], [], 0.0
    for jid in manifest["jobs"]:
        job = service.job(jid)
        status = str(job.status())
        if status not in ("DONE", "JobStatus.DONE"):
            print(f"[collect] job {jid} is {status}; try again later")
            return 1
        res = list(job.result())
        pubs += res
        job_of += [jid] * len(res)
        usage += float((job.metrics().get("usage") or {}).get("quantum_seconds") or 0)
    rows = score_rows(manifest, pubs)
    for r, jid in zip(rows, job_of):
        r["job_id"] = jid
    write_csv(path.with_suffix(".csv"), rows)
    summarize(rows)
    print(f"[collect] {len(manifest['jobs'])} jobs, {usage:.0f} QPU seconds used")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="study", required=True)
    for name, (_, n_default, p_default) in STUDIES.items():
        s = sub.add_parser(name)
        s.add_argument("--backend", default="fake_kawasaki", help="ibm_<device> or fake_<device>")
        s.add_argument("--instance", help="account instance, e.g. the QII one")
        s.add_argument("--n", nargs="*", type=int, default=n_default)
        s.add_argument("--p", nargs="*", type=int, default=p_default)
        s.add_argument("--families", nargs="*")
        s.add_argument("--seeds", nargs="*", type=int, default=[0])
        s.add_argument("--shots", type=int, default=4000)
        s.add_argument("--per-job", type=int, default=100, help="PUBs per job")
        s.add_argument("--min-fid", type=float, default=0.01, help="drop circuits below this")
        s.add_argument("--max-2q", type=int, help="drop circuits with more two-qubit gates")
        s.add_argument("--no-fractional", action="store_true", help="Heron: cz only, no rzz")
        s.add_argument("--no-dd", action="store_true", help="no dynamical decoupling")
        s.add_argument("--twirl", action="store_true", help="gate and readout twirling")
        s.add_argument("--mode", choices=["batch", "job"], default="batch")
        s.add_argument("--submit", action="store_true", help="really send the jobs")
        s.add_argument("--simulate", action="store_true", help="local Aer, live calibration")
        s.add_argument("--noiseless", action="store_true", help="same ISA circuits, no noise")
        s.add_argument("--dry-run", action="store_true", help="transpile and estimate only")
    c = sub.add_parser("collect")
    c.add_argument("manifest")
    c.add_argument("--instance")
    a = ap.parse_args()
    return collect(a) if a.study == "collect" else run(a)


if __name__ == "__main__":
    sys.exit(main())
