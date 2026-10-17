"""IBM Quantum side: device circuits, transpiling, readout correction, runs.

The target is ibm_kawasaki, the IBM Quantum System One the University of Tokyo
runs for the QII consortium. It was a 127-qubit Eagle r3 (ECR) and is due a
156-qubit Heron (CZ, fractional rzz), so nothing here assumes either: gates,
couplings and errors all come from the device's Target.

Clbit v always holds vertex v, whatever the routing did, so int(bitstring, 2)
indexes Problem.cuts() directly.
"""

import math

import numpy as np
import scipy.sparse as sp
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler import generate_preset_pass_manager
from scipy.sparse.csgraph import reverse_cuthill_mckee

DEVICE = "ibm_kawasaki"
TWO_Q = ("rzz", "cz", "ecr", "cx")
DENSITY_MAX_N = 14  # noisy local runs: density matrix up to here (4 GB at fp64)


def account(instance=None):
    from qiskit_ibm_runtime import QiskitRuntimeService

    try:
        return QiskitRuntimeService(instance=instance)
    except Exception as e:  # no saved account, wrong instance, offline
        raise SystemExit(f"no IBM Quantum account ({e}); save one first, see README") from e


def backend(name=DEVICE, fractional=True, instance=None):
    """backend() - a device, or fake_<device> for Qiskit's shipped snapshot of it
    @name: 'ibm_kawasaki', 'fake_kawasaki', ...
    @fractional: ask for fractional gates (Heron only)
    @instance: account instance to go through, the saved default if None
    """
    if name.startswith("fake_"):
        from qiskit_ibm_runtime import fake_provider

        return getattr(fake_provider, "Fake" + name[5:].title().replace("_", ""))()
    service = account(instance)
    if fractional:
        try:
            return service.backend(name, use_fractional_gates=True)
        except Exception:  # Eagle has none
            pass
    return service.backend(name)


def processor(bk):
    pt = getattr(bk, "processor_type", None) or {}
    return f"{pt.get('family', '?')} r{pt.get('revision', '?')}" if pt else "?"


def _edges(P):
    w = {}
    for i, j, x in zip(P.src.tolist(), P.tgt.tolist(), P.wn.tolist()):
        key = (min(i, j), max(i, j))
        w[key] = w.get(key, 0.0) + x
    return w


def _empty(n, m=None):
    return QuantumCircuit(QuantumRegister(n, "q"), ClassicalRegister(n if m is None else m, "meas"))


def logical(P, gammas, betas):
    qc = _empty(P.n)
    qc.h(range(P.n))
    for g, b in zip(gammas, betas):
        for (i, j), x in _edges(P).items():
            qc.rzz(2.0 * float(g) * x, i, j)
        qc.rx(-2.0 * float(b), range(P.n))
    qc.measure(range(P.n), range(P.n))
    return qc


def line_circuit(P, gammas, betas):
    """line_circuit() - the QAOA circuit on a line, edges brought together by swaps
    @P: the Problem
    @gammas: cost angles, one per layer
    @betas: mixer angles, one per layer

    pos[i] is the vertex at line position i, starting from a reverse
    Cuthill-McKee order so sparse graphs need few swap layers. A cost layer
    runs brick-wall layers (0,1)(2,3).. then (1,2)(3,4)..: each pair gets its
    rzz if it is an edge still to do, then swaps. n layers reverse the line
    and meet every pair once, and the layer stops as soon as the last edge is
    in. All ZZ terms commute, so the order is exact.
    """
    n, w = P.n, _edges(P)
    adj = sp.coo_matrix((np.ones(len(w)), tuple(zip(*w))), shape=(n, n)).tocsr() if w else None
    pos = list(reverse_cuthill_mckee(adj + adj.T)) if w else list(range(n))
    qc = _empty(n)
    qc.h(range(n))
    for g, b in zip(gammas, betas):
        todo, layer = dict(w), 0
        while todo:
            if layer > n:
                raise RuntimeError("swap network left edges out")
            for i in range(layer % 2, n - 1, 2):
                key = (min(pos[i], pos[i + 1]), max(pos[i], pos[i + 1]))
                if key in todo:
                    qc.rzz(2.0 * float(g) * todo.pop(key), i, i + 1)
                if todo:
                    qc.swap(i, i + 1)
                    pos[i], pos[i + 1] = pos[i + 1], pos[i]
            layer += 1
        qc.rx(-2.0 * float(b), range(n))
    qc.measure(range(n), pos)
    return qc


def _errors(target):
    # Best error per coupled pair over all two-qubit gates. Fractional rzz may
    # come uncalibrated, and Eagle's ECR is directed, so a pair shows up once.
    two = {}
    for gate in (g for g in TWO_Q if g in target.operation_names):
        for (a, b), props in target[gate].items():
            if props is None or props.error is None:
                continue
            key = (min(a, b), max(a, b))
            two[key] = min(two.get(key, 1.0), props.error)
    ro = {
        q[0]: (p.error if p is not None and p.error is not None else 1.0)
        for q, p in target["measure"].items()
    }
    return two, ro


def best_line(target, n):
    """best_line() - n physical qubits forming a path, fewest expected errors
    @target: the device's Target
    @n: path length

    Cost is -sum log(1 - e) over the path's two-qubit gates and readouts;
    depth-first search, pruned against the best path so far.
    """
    two, ro = _errors(target)
    adj = {}
    for a, b in two:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    def nll(e):
        return -math.log(max(1e-12, 1.0 - e)) if e < 1.0 else math.inf

    best = [math.inf, None]

    def walk(path, cost):
        if cost >= best[0]:
            return
        if len(path) == n:
            best[:] = [cost, list(path)]
            return
        last = path[-1]
        for q in adj.get(last, ()):
            if q not in path:
                path.append(q)
                walk(path, cost + nll(two[(min(last, q), max(last, q))]) + nll(ro.get(q, 1.0)))
                path.pop()

    for q in sorted(adj, key=lambda q: ro.get(q, 1.0)):
        walk([q], nll(ro.get(q, 1.0)))
    if best[1] is None:
        raise RuntimeError(f"no path of {n} working qubits")
    return best[1]


def count_2q(qc):
    ops = qc.count_ops()
    return sum(ops.get(g, 0) for g in TWO_Q)


def transpile(P, gammas, betas, target, line=None, router="best", seed=0):
    """transpile() - ISA circuit for @target
    @P: the Problem
    @gammas: cost angles, one per layer
    @betas: mixer angles, one per layer
    @target: the device's Target
    @line: physical path for the swap network, best_line() if None
    @router: 'swap', 'sabre' or 'best' (both, higher fidelity() wins)
    @seed: transpiler seed

    The swap layout keeps the level 1 init stage, since level 3 would elide
    the network's swaps and route everything again. On Eagle's uneven
    couplers fewer gates is not always fewer errors, hence fidelity().

    Return: (ISA circuit, router used).
    """
    out = {}
    if router in ("swap", "best"):
        line = line or best_line(target, P.n)
        pm = generate_preset_pass_manager(
            optimization_level=3, target=target, initial_layout=line, seed_transpiler=seed
        )
        pm.init = generate_preset_pass_manager(optimization_level=1, target=target).init
        out["swap"] = pm.run(line_circuit(P, gammas, betas))
    if router in ("sabre", "best"):
        pm = generate_preset_pass_manager(optimization_level=3, target=target, seed_transpiler=seed)
        out["sabre"] = pm.run(logical(P, gammas, betas))
    out = {k: _fold_phase_gates(qc) for k, qc in out.items()}
    name = max(out, key=lambda k: (fidelity(out[k], target), -count_2q(out[k])))
    return out[name], name


def _fold_phase_gates(qc):
    # Wrapping rzz angles into [0, pi/2] leaves global_phase gates behind,
    # and no device takes them.
    if not any(i.operation.name == "global_phase" for i in qc.data):
        return qc
    out = qc.copy_empty_like()
    for inst in qc.data:
        if inst.operation.name == "global_phase":
            out.global_phase += inst.operation.params[0]
        else:
            out.append(inst)
    return out


def fidelity(isa, target):
    """fidelity() - odds that @isa runs without a single gate or readout error
    @isa: an ISA circuit
    @target: the device's Target

    Idle decoherence is left out. On the Fez and Kawasaki models the share of
    P(x*) a circuit keeps comes out a little above it, about F^0.7.
    """
    log_f = 0.0
    for inst in isa.data:
        name = inst.operation.name
        if name not in target.operation_names:
            continue
        props = target[name].get(tuple(isa.find_bit(q).index for q in inst.qubits))
        if props is not None and props.error is not None:
            log_f += math.log(max(1e-12, 1.0 - props.error))
    return math.exp(log_f)


def measured_qubits(isa):
    """Physical qubit read into each clbit."""
    m = {
        isa.find_bit(i.clbits[0]).index: isa.find_bit(i.qubits[0]).index
        for i in isa.data
        if i.operation.name == "measure"
    }
    return tuple(m[c] for c in range(isa.num_clbits))


def calibration(qubits, num_qubits):
    """The all-0 and all-1 readout circuits; clbit k reads qubits[k]."""
    out = []
    for flip in (False, True):
        qc = _empty(num_qubits, len(qubits))
        if flip:
            qc.x(list(qubits))
        qc.measure(list(qubits), range(len(qubits)))
        out.append(qc)
    return out


def flip_rates(bits0, bits1):
    """Per-clbit P(read 1 | 0) and P(read 0 | 1) from the calibration shots."""
    return bits0.to_bool_array("little").mean(0), 1.0 - bits1.to_bool_array("little").mean(0)


def unfold(dist, e01, e10):
    """unfold() - undo independent readout flips on a 2^n distribution
    @dist: probabilities indexed by int(bitstring, 2)
    @e01: per-bit P(read 1 | 0)
    @e10: per-bit P(read 0 | 1)

    Tensored inverse; bit j is axis n-1-j once reshaped. The result still sums
    to 1 but may dip below 0 here and there.
    """
    n = len(e01)
    q = dist.reshape((2,) * n)
    for j in range(n):
        a = np.array([[1.0 - e01[j], e10[j]], [e01[j], 1.0 - e10[j]]])
        q = np.moveaxis(np.tensordot(np.linalg.inv(a), q, axes=([1], [n - 1 - j])), 0, n - 1 - j)
    return q.reshape(-1)


def score(P, bits, rates=None):
    """score() - P(x*), mean cut ratio and expected cut from sampled bits
    @P: the Problem
    @bits: a BitArray (pub_result.data.meas)
    @rates: (e01, e10) from flip_rates(), adds the readout-corrected values
    """
    counts = bits.get_int_counts()
    idx = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
    c = np.fromiter(counts.values(), dtype=float, count=len(counts))
    shots = c.sum()
    cuts, opt = P.cuts(), P.optimal_indices()
    ec = float(c @ cuts[idx]) / shots
    sd = math.sqrt(max(0.0, float(c @ cuts[idx] ** 2) / shots - ec**2))
    out = dict(
        success_probability=float(c[np.isin(idx, opt)].sum()) / shots,
        mean_ratio=ec / P.best,
        mean_ratio_err=sd / math.sqrt(shots) / abs(P.best),
        expected_cut=ec,
        shots=int(shots),
    )
    if rates is not None:
        q = unfold(np.bincount(idx, weights=c, minlength=2**P.n) / shots, *rates)
        out.update(
            success_probability_mitigated=float(q[opt].sum()),
            mean_ratio_mitigated=float(q @ cuts) / P.best,
            readout_error_mean=float(np.mean(rates[0] + rates[1]) / 2),
        )
    return out


def qpu_seconds(isa, target, shots, rep_delay=250e-6):
    try:
        t = isa.estimate_duration(target, unit="s")
    except Exception:
        t = 0.0
    return shots * (t + rep_delay)


def width(isa):
    # SABRE may route through spare qubits, so this can exceed the qubits read
    skip = ("barrier", "delay")
    return len(
        {isa.find_bit(q).index for i in isa.data if i.operation.name not in skip for q in i.qubits}
    )


def sim_method(n, noiseless=False):
    # Shot trajectories above DENSITY_MAX_N: 4000 noisy shots on 20 qubits take
    # about 3 min on an RTX 3080. Only the readout calibrations go past 24, and
    # they are product states.
    if n > 24:
        return "matrix_product_state"
    return "density_matrix" if not noiseless and n <= DENSITY_MAX_N else "statevector"


def simulator(bk, n, noiseless=False):
    """simulator() - local Aer sampler for circuits of @bk
    @bk: fake_* snapshot, or a real device for its live calibration
    @n: the circuit's width()
    @noiseless: ideal gates and readout
    """
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2

    method = sim_method(n, noiseless)
    kw = dict(method=method)
    if method != "matrix_product_state" and "GPU" in AerSimulator().available_devices():
        kw.update(device="GPU", **(dict(cuStateVec_enable=True) if method == "statevector" else {}))
    sim = AerSimulator(**kw) if noiseless else AerSimulator.from_backend(bk, **kw)
    return SamplerV2.from_backend(sim)


def sampler(mode, dd=True, twirl=False):
    from qiskit_ibm_runtime import SamplerV2

    s = SamplerV2(mode=mode)
    s.options.dynamical_decoupling.enable = dd
    if dd:
        s.options.dynamical_decoupling.sequence_type = "XpXm"
    s.options.twirling.enable_gates = twirl
    s.options.twirling.enable_measure = twirl
    return s
