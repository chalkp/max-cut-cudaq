"""MaxCut instances, make_graph(family, n, seed).

  ring                  cycle C_n, unit weights (solved exactly in exact.py)
  bipartite             K(n/2, n/2), unit weights
  dense                 K_n, integer weights in [-10, 10]
  ramanujan             random 3-regular, Ramanujan certified, unit weights
  clustered             two cliques joined by 6 bridges, weights in [1, 10]
  euclidean             K_n on random points, weights ceil(10^4 * distance)
  erdos_renyi           connected G(n, 0.7), weights in [1, 10]
  heterogeneous_weight  Barabasi-Albert BA(n, n/2), weights in [-10, 10]
  irregular             complete, cycle, star and path blocks in a chain, unit weights

Don't reorder the RNG calls: every (family, n, seed) must keep giving the same
graph, or the sweep results stop being comparable.
"""

import math

import networkx as nx
import numpy as np


def _integer_weights(G, low, high, seed):
    w = np.random.default_rng(seed).integers(low, high + 1, size=G.number_of_edges())
    for (u, v), x in zip(G.edges(), w):
        G.edges[u, v]["weight"] = float(x)
    return G


def _unit(G):
    nx.set_edge_attributes(G, values=1.0, name="weight")
    return G


def ring(n, seed=0):
    if n < 3:
        raise ValueError(f"a ring needs n >= 3, got {n}")
    return _unit(nx.cycle_graph(n))


def bipartite(n, seed=0):
    return _unit(nx.complete_bipartite_graph(n // 2, n - n // 2))


def dense(n, seed=0):
    return _integer_weights(nx.complete_graph(n), -10, 10, seed)


def heterogeneous_weight(n, seed=0):
    G = nx.barabasi_albert_graph(n, max(1, n // 2), seed=np.random.default_rng(seed))
    return _integer_weights(G, -10, 10, seed)


def irregular(n, seed=0, n_blocks=4):
    if n < 3 * n_blocks:
        raise ValueError(f"irregular needs n >= {3 * n_blocks}, got {n}")
    rng = np.random.default_rng(seed)
    motifs = (nx.complete_graph, nx.cycle_graph, lambda s: nx.star_graph(s - 1), nx.path_graph)
    sizes, remaining = [], n
    for i in range(n_blocks):
        if i == n_blocks - 1:
            size = remaining
        else:
            size = int(rng.integers(3, max(3, remaining - 3 * (n_blocks - i - 1)) + 1))
        sizes.append(size)
        remaining -= size
    G, offsets = motifs[0](sizes[0]), [0]
    for i, size in enumerate(sizes[1:], start=1):
        offsets.append(G.number_of_nodes())
        G = nx.disjoint_union(G, motifs[i % 4](size))
    for i in range(n_blocks - 1):
        u = rng.integers(0, sizes[i]) + offsets[i]
        v = rng.integers(0, sizes[i + 1]) + offsets[i + 1]
        G.add_edge(int(u), int(v))
    return _unit(G)


def _second_eigenvalue(G, degree):
    if not nx.is_connected(G):
        return None
    ev = np.linalg.eigvalsh(nx.to_numpy_array(G))
    rest = [v for v in ev if abs(abs(v) - degree) > 1e-8]
    return max(abs(v) for v in rest) if rest else 0.0


def ramanujan(n, seed=0, degree=3, max_tries=2000):
    if n * degree % 2 or n <= degree:
        raise ValueError(f"no {degree}-regular graph on {n} nodes")
    bound = 2.0 * np.sqrt(degree - 1)
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        G = nx.random_regular_graph(degree, n, seed=int(rng.integers(0, 2**31 - 1)))
        lam = _second_eigenvalue(G, degree)
        if lam is not None and lam <= bound + 1e-9:
            return _unit(G)
    raise RuntimeError(f"no Ramanujan {degree}-regular graph on {n} nodes in {max_tries} tries")


def clustered(n, seed=0, n_blocks=2, n_bridges=6):
    if n < 4 * n_blocks:
        raise ValueError(f"clustered needs n >= {4 * n_blocks}, got {n}")
    rng = np.random.default_rng(seed)
    sizes = [n // n_blocks] * n_blocks
    for i in range(n - sum(sizes)):
        sizes[i] += 1
    G, starts, offset = nx.Graph(), [], 0
    for size in sizes:
        starts.append(offset)
        G = nx.disjoint_union(G, nx.complete_graph(size)) if offset else nx.complete_graph(size)
        offset += size
    for i in range(n_blocks - 1):
        eu = rng.choice(sizes[i], size=min(n_bridges, sizes[i]), replace=False)
        ev = rng.choice(sizes[i + 1], size=min(n_bridges, sizes[i + 1]), replace=False)
        for a, b in zip(eu, ev):
            G.add_edge(int(a) + starts[i], int(b) + starts[i + 1])
    return _integer_weights(G, 1, 10, seed)


def euclidean(n, seed=0, scale=10_000):
    points = np.random.default_rng([seed, n]).random((n, 2))
    G = nx.complete_graph(n)
    for u, v in G.edges():
        d = float(np.linalg.norm(points[u] - points[v]))
        # ceil, not round: keeps the triangle inequality
        G.edges[u, v]["weight"] = float(max(1, math.ceil(scale * d)))
    return G


def erdos_renyi(n, seed=0, density=0.7):
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        G = nx.gnp_random_graph(n, density, seed=int(rng.integers(0, 2**31 - 1)))
        if nx.is_connected(G):
            return _integer_weights(G, 1, 10, seed)
    raise RuntimeError(f"no connected G({n}, {density}) in 1000 tries")


FAMILIES = {
    "ring": ring,
    "bipartite": bipartite,
    "dense": dense,
    "ramanujan": ramanujan,
    "clustered": clustered,
    "euclidean": euclidean,
    "erdos_renyi": erdos_renyi,
    "heterogeneous_weight": heterogeneous_weight,
    "irregular": irregular,
}
MIN_N = {"irregular": 12, "clustered": 8}

LABELS = {
    "bipartite": "complete bipartite  K(n/2, n/2)",
    "dense": "dense  K(n), weights ±10",
    "heterogeneous_weight": "heterogeneous weight  BA(n, n/2)",
    "irregular": "irregular  4 mixed motifs",
    "ring": "ring of disagrees  C(n)",
    "ramanujan": "Ramanujan  3-regular expander",
    "clustered": "clustered  2 blocks + bridge",
    "euclidean": "Euclidean  complete, metric weights",
    "erdos_renyi": "Erdos-Renyi  G(n, 0.7)",
}


def make_graph(family, n, seed=0):
    return FAMILIES[family](n, seed)


def special_graph(kind, n, seed=0):
    """Named graphs for the Q6 threshold runs; anything else goes to make_graph()."""
    if kind == "ring":
        return _unit(nx.cycle_graph(n))
    if kind == "hypercube":
        return _unit(nx.convert_node_labels_to_integers(nx.hypercube_graph(int(np.log2(n)))))
    if kind == "complete bipartite":
        return _unit(nx.complete_bipartite_graph(n // 2, n // 2))
    if kind == "Petersen":
        return _unit(nx.petersen_graph())
    if kind == "torus 4x4":
        return _unit(nx.convert_node_labels_to_integers(nx.grid_2d_graph(4, 4, periodic=True)))
    return make_graph(kind, n, seed)


def edge_arrays(G):
    """(src, tgt, w) arrays, nodes numbered in G.nodes() order."""
    idx = {u: i for i, u in enumerate(G.nodes())}
    e = [(idx[u], idx[v], float(d.get("weight", 1.0))) for u, v, d in G.edges(data=True)]
    return (
        np.array([a for a, _, _ in e], dtype=np.int64),
        np.array([b for _, b, _ in e], dtype=np.int64),
        np.array([c for _, _, c in e], dtype=float),
    )


def weighted_complete(n, seed, kind):
    """weighted_complete() - K_n with random weights, for the Q5/Q6 runs
    @n: number of nodes
    @seed: instance seed
    @kind: one of
        'signed'    integers in [-10, 10], 0 -> 1          rng(1000 n + seed)
        'positive'  from {0.1, 0.2, 0.3, 0.5, 1.0}         rng(1000 n + seed)
        'uniform'   integers 1..1000, / 1000               rng(7000 n + seed)
        'hardware'  from {0.1, 0.2, 0.3, 0.5, 1.0}         rng(7000 n + seed)

    Return: (src, tgt, w).
    """
    iu = np.triu_indices(n, 1)
    if kind in ("signed", "positive"):
        rng = np.random.default_rng(1000 * n + seed)
        if kind == "signed":
            w = rng.integers(-10, 11, size=iu[0].size).astype(float)
            w[w == 0] = 1.0
        else:
            w = rng.choice([0.1, 0.2, 0.3, 0.5, 1.0], size=iu[0].size)
    else:
        rng = np.random.default_rng(7000 * n + seed)
        if kind == "uniform":
            w = rng.integers(1, 1001, size=iu[0].size) / 1000.0
        else:
            w = rng.choice([0.1, 0.2, 0.3, 0.5, 1.0], size=iu[0].size)
    return iu[0].astype(np.int64), iu[1].astype(np.int64), w
