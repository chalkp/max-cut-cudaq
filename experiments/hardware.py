"""LR-QAOA on IBM Quantum hardware, scored against the exact simulator.

  sweep           LR ramp, Delta = (0.6, 0.3), P(x*) and cut ratio against p
  pilot           linear vs gap-adapted vs the two cubic schedules
  depth-matched   LR against COBYLA angles optimised on the exact state

Each cell (instance, angles) is one ISA circuit and one PUB. A fake_* backend
runs at once on its local noise model. A real device only gets an estimate
unless --submit is given; the jobs then go out in one Batch and their ids land
in results/hardware/<study>_<device>.json as they leave, for `collect`.

    hardware.py sweep --backend fake_fez --n 8 10 --p 1 2 3 5
    hardware.py sweep --backend fake_fez --noiseless    # same circuits, ideal
    hardware.py sweep --backend ibm_fez                 # estimate only
    hardware.py sweep --backend ibm_fez --submit
    hardware.py collect results/hardware/sweep_ibm_fez.json
"""

import argparse
import csv
import datetime
import json
import pathlib
import sys
import time

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


STUDIES = {
    "sweep": (sweep_cells, [8, 10, 12, 14, 16], [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]),
    "pilot": (pilot_cells, [8, 10, 12], [5, 10, 20]),
    "depth-matched": (depth_matched_cells, [8, 10, 12], [10, 20]),
}


def build(a, backend):
    """Cells, their ISA circuits and the exact references, in PUB order."""
    make, *_ = STUDIES[a.study]
    target, lines, cells, isas = backend.target, {}, [], []
    t0 = time.time()
    for meta, (g, b) in make(a):
        P = problem(meta["family"], meta["n"], meta["seed"])
        if meta["n"] not in lines:
            lines[meta["n"]] = ibm.best_line(target, meta["n"])
        isa, router = ibm.transpile(P, g, b, target, line=lines[meta["n"]])
        n2q = ibm.count_2q(isa)
        if a.max_2q and n2q > a.max_2q:
            continue
        ref = P.evaluate(g, b)
        meta.update(
            gammas=[float(x) for x in g],
            betas=[float(x) for x in b],
            router=router,
            n_2q=n2q,
            depth=isa.depth(),
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
            f"P*exact={ref['success_probability']:.4f}",
            flush=True,
        )
    print(f"[build] {len(cells)} circuits in {time.time() - t0:.0f} s", flush=True)
    return cells, isas


def score_rows(manifest, results):
    """results: one SamplerV2 result per job, PUBs in cell order."""
    rows, k = [], 0
    for job, res in zip(manifest["jobs"], results):
        for pub in res:
            c = manifest["cells"][k]
            k += 1
            r = ibm.score(problem(c["family"], c["n"], c["seed"]), pub.data.meas)
            row = {key: v for key, v in c.items() if key not in ("gammas", "betas")}
            row.update(r, backend=manifest["backend"], job_id=job, created=manifest["created"])
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
    d["P_hw/P_exact"] = d.success_probability / d.success_probability_exact
    cols = ["success_probability", "success_probability_exact", "mean_ratio", "mean_ratio_exact"]
    keys = ["n", "schedule"] if d.schedule.nunique() > 1 else ["n", "p"]
    print(d.groupby(keys)[cols + ["P_hw/P_exact"]].mean().round(4).to_string())


def run(a):
    bk = ibm.backend(a.backend, fractional=not a.no_fractional)
    fractional = "rzz" in bk.target.operation_names
    print(f"[{a.study}] {bk.name}: {bk.num_qubits} qubits, fractional rzz {fractional}", flush=True)
    cells, isas = build(a, bk)
    qpu = sum(c["qpu_s"] for c in cells)
    print(f"[estimate] {len(cells)} circuits x {a.shots} shots, ~{qpu / 60:.1f} QPU minutes")
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"{a.study.replace('-', '_')}_{bk.name}{'_ideal' if a.noiseless else ''}"
    manifest = dict(
        study=a.study,
        backend=bk.name,
        fractional=fractional,
        shots=a.shots,
        dd=not a.no_dd,
        twirl=a.twirl,
        noiseless=a.noiseless,
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        cells=cells,
        jobs=[],
    )
    chunks = [range(i, min(i + a.per_job, len(isas))) for i in range(0, len(isas), a.per_job)]
    if a.dry_run:
        print("[dry run] nothing run or submitted")
        return 0
    if a.noiseless and not bk.name.startswith("fake"):
        raise SystemExit("--noiseless needs a fake_* backend")
    if bk.name.startswith("fake"):
        # XXX: one noisy Aer run holding several ISA circuits can return zeros
        # for all but the first (qiskit-aer 0.17, CPU). Real devices are fine,
        # so run local PUBs one at a time.
        if a.noiseless:
            from qiskit_aer.primitives import SamplerV2 as AerSampler

            s = AerSampler()
        else:
            s = ibm.sampler(bk)
        results = [[s.run([isa], shots=a.shots).result()[0] for isa in isas]]
        manifest["jobs"].append("local")
        stem.with_suffix(".json").write_text(json.dumps(manifest, indent=1))
        rows = score_rows(manifest, results)
        write_csv(stem.with_suffix(".csv"), rows)
        summarize(rows)
        return 0
    if not a.submit:
        print("[dry run] nothing submitted; add --submit to send these circuits")
        return 0
    from qiskit_ibm_runtime import Batch

    path = stem.with_suffix(".json")
    with Batch(backend=bk) as batch:
        s = ibm.sampler(batch, dd=not a.no_dd, twirl=a.twirl)
        for ch in chunks:
            job = s.run([isas[i] for i in ch], shots=a.shots)
            manifest["jobs"].append(job.job_id())
            path.write_text(json.dumps(manifest, indent=1))
            print(f"[submit] job {job.job_id()}: {len(ch)} circuits", flush=True)
    print(f"[submit] {len(chunks)} jobs; collect with: python {sys.argv[0]} collect {path}")
    return 0


def collect(a):
    from qiskit_ibm_runtime import QiskitRuntimeService

    path = pathlib.Path(a.manifest)
    manifest = json.loads(path.read_text())
    service = QiskitRuntimeService()
    results, usage = [], 0.0
    for jid in manifest["jobs"]:
        job = service.job(jid)
        status = str(job.status())
        if status not in ("DONE", "JobStatus.DONE"):
            print(f"[collect] job {jid} is {status}; try again later")
            return 1
        results.append(job.result())
        usage += float((job.metrics().get("usage") or {}).get("quantum_seconds") or 0)
    rows = score_rows(manifest, results)
    write_csv(path.with_suffix(".csv"), rows)
    summarize(rows)
    print(f"[collect] {len(manifest['jobs'])} jobs, {usage:.0f} QPU seconds used")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="study", required=True)
    for name, (_, n_default, p_default) in STUDIES.items():
        s = sub.add_parser(name)
        s.add_argument("--backend", default="fake_fez", help="ibm_<device> or fake_<device>")
        s.add_argument("--n", nargs="*", type=int, default=n_default)
        s.add_argument("--p", nargs="*", type=int, default=p_default)
        s.add_argument("--families", nargs="*")
        s.add_argument("--seeds", nargs="*", type=int, default=[0])
        s.add_argument("--shots", type=int, default=4000)
        s.add_argument("--per-job", type=int, default=100, help="circuits per job")
        s.add_argument("--max-2q", type=int, help="skip circuits with more two-qubit gates")
        s.add_argument("--no-fractional", action="store_true", help="plain cz, no native rzz")
        s.add_argument("--no-dd", action="store_true", help="no dynamical decoupling")
        s.add_argument("--twirl", action="store_true", help="gate and readout twirling")
        s.add_argument("--submit", action="store_true", help="really send the jobs")
        s.add_argument("--dry-run", action="store_true", help="transpile and estimate only")
        s.add_argument(
            "--noiseless", action="store_true", help="fake_* only: same ISA circuits, no noise"
        )
    c = sub.add_parser("collect")
    c.add_argument("manifest")
    a = ap.parse_args()
    return collect(a) if a.study == "collect" else run(a)


if __name__ == "__main__":
    sys.exit(main())
