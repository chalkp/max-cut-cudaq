"""IBM Quantum side: device circuits, transpiling, runs.

Clbit v always holds vertex v, whatever the routing did, so int(bitstring, 2)
indexes Problem.cuts() directly.
"""

import math

import numpy as np
import scipy.sparse as sp
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler import generate_preset_pass_manager
from scipy.sparse.csgraph import reverse_cuthill_mckee

TWO_Q = ("rzz", "cz", "ecr", "cx")


def _edges(P):
    w = {}
    for i, j, x in zip(P.src.tolist(), P.tgt.tolist(), P.wn.tolist()):
        key = (min(i, j), max(i, j))
        w[key] = w.get(key, 0.0) + x
    return w


def _empty(n):
    return QuantumCircuit(QuantumRegister(n, "q"), ClassicalRegister(n, "meas"))


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
    @router: 'swap', 'sabre' or 'best' (both, fewer two-qubit gates wins)
    @seed: transpiler seed

    The swap layout keeps the level 1 init stage, since level 3 would elide
    the network's swaps and route everything again.

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
    name = min(out, key=lambda k: (count_2q(out[k]), out[k].depth()))
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
