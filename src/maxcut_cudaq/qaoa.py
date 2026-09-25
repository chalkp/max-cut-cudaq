r"""LR-QAOA (or any other angle schedule) for MaxCut, with exact readout.

    |psi> = prod_k e^{+i beta_k sum X} e^{-i gamma_k H_C} |+>^n
    H_C   = sum_ij (w_ij / max|w|) Z_i Z_j

The mixer sign follows the LR-QAOA paper (their R_X(-2 beta)). The cost layer is
built from diagonal gates,

    e^{-i t Z_i Z_j} = e^{-i t} r1(2t)_i r1(2t)_j cr1(-4t)_ij

with the r1 angles summed per qubit, so one layer is |E| cr1 + n r1 + n rx. On
n >= 24 that is 3-4x faster than cx-rz-cx, and exp_pauli never gets fused.

Each instance picks a backend (MAXCUT_BACKEND=auto|cudaq|numpy):
 - numpy below ~13 qubits on dense graphs; CUDA-Q costs 3-8 us per gate there
   and a 2^12 state is too small to win that back.
 - CUDA-Q otherwise: nvidia-legacy (plain cuStateVec, fp64) up to 17 qubits,
   nvidia (cusvsim) above, fp64 unless set_precision('fp32').

Nothing is sampled. P(x*) counts every optimal bitstring, degenerate ones too.
"""

import os

import numpy as np

from .cpu import hc_vector, probabilities as numpy_probabilities

DG, DB = 0.6, 0.3
NUMPY_NS_PER_AMP = 4.0  # one core, per amplitude per qubit per layer
CUDAQ_US_PER_GATE = 3.5  # nvidia-legacy at small n, host bound

_PRECISION = os.environ.get("MAXCUT_PRECISION", "fp64")


def set_precision(precision):
    """set_precision() - pick fp32 or fp64 for circuits above 17 qubits
    @precision: 'fp64' (default) or 'fp32'

    fp32 is 2x faster on dense graphs and ~10x on sparse ones at 24-28 qubits;
    P(x*) stays within 5e-5 relative after 300 layers. Applies from the next
    circuit on. The fusion settings are fixed when cudaq loads, see engine.py.
    """
    global _PRECISION
    _PRECISION = precision


def precision():
    return _PRECISION


def angles(p, dg=DG, db=DB, u=None):
    """Linear ramp, or schedule u[0..p]: gamma_k = u[k+1] dg, beta_k = (1 - u[k]) db."""
    u = np.linspace(0.0, 1.0, p + 1) if u is None else np.asarray(u, dtype=float)
    return u[1:] * dg, (1 - u[:-1]) * db


def choose_backend(n, n_edges):
    numpy_cost = NUMPY_NS_PER_AMP * 1e-3 * 2**n * n  # us per layer
    cudaq_cost = CUDAQ_US_PER_GATE * (n_edges + 2 * n)
    return "numpy" if numpy_cost < cudaq_cost else "cudaq"


class Problem:
    """One MaxCut instance.

    Cuts are scored with the original weights @w; the circuit sees w / max|w|.
    @backend is 'auto' (default), 'cudaq' or 'numpy'.
    """

    def __init__(self, src, tgt, w, n=None, rtol=1e-9, backend=None):
        self.src = np.asarray(src, dtype=np.int64)
        self.tgt = np.asarray(tgt, dtype=np.int64)
        self.w = np.asarray(w, dtype=float)
        self.n = int(n if n is not None else max(self.src.max(), self.tgt.max()) + 1)
        self.scale = float(np.abs(self.w).max()) or 1.0
        self.wn = self.w / self.scale
        deg = np.zeros(self.n)
        np.add.at(deg, self.src, self.wn)
        np.add.at(deg, self.tgt, self.wn)
        self.deg = deg
        self.rtol = rtol
        backend = backend or os.environ.get("MAXCUT_BACKEND", "auto")
        self.backend = choose_backend(self.n, len(self.w)) if backend == "auto" else backend
        self._dev = self._host = None
        self.best = self.threshold = None

    @classmethod
    def from_graph(cls, G, **kw):
        from .graphs import edge_arrays

        return cls(*edge_arrays(G), n=G.number_of_nodes(), **kw)

    def _set_best(self, best):
        self.best = float(best)
        self.threshold = self.best - self.rtol * max(1.0, abs(self.best))

    def cut_device(self):
        """Cut value of every bitstring, on the GPU."""
        if self._dev is None:
            from . import gpu

            self._dev = gpu.cut_values(self.n, self.src, self.tgt, self.w)
            self._set_best(self._dev.max())
        return self._dev

    def _host_arrays(self):
        # (cut, optimal mask, H_C). The numpy backend builds them on the CPU so
        # it never creates a CUDA context.
        if self._host is None:
            if self.backend == "numpy":
                x = np.arange(2**self.n)
                cut = (((x[:, None] >> self.src) ^ (x[:, None] >> self.tgt)) & 1) @ self.w
                self._set_best(cut.max())
            else:
                cut = self.cut_device().get()
            self._host = (cut, cut >= self.threshold, float(self.wn.sum()) - 2.0 / self.scale * cut)
        return self._host

    def optimum(self):
        if self.best is None:
            self._host_arrays() if self.backend == "numpy" else self.cut_device()
        return self.best

    def cut_host(self):
        return self._host_arrays()[0].copy()

    def optimal_mask_host(self):
        return self._host_arrays()[1].copy()

    def hc_host(self):
        """H_C with normalized weights, i.e. sum wn - 2 cut / max|w|."""
        return self._host_arrays()[2].copy()

    def evaluate(self, gammas, betas):
        """evaluate() - run one circuit and score it
        @gammas: cost angles, one per layer
        @betas: mixer angles, one per layer

        Return: dict with success_probability, mean_ratio and expected_cut.
        """
        if self.backend == "numpy":
            cut, opt, hc = self._host_arrays()
            pr = numpy_probabilities(self.n, hc, gammas, betas)
            ps, ec = float(pr[opt].sum()), float(pr @ cut)
        else:
            from . import engine

            ps, ec = engine.evaluate(self, gammas, betas, _PRECISION)
        return dict(
            success_probability=ps,
            mean_ratio=ec / self.best if self.best else float("nan"),
            expected_cut=ec,
        )

    def lr(self, p, dg=DG, db=DB, u=None):
        return self.evaluate(*angles(p, dg, db, u))

    def probabilities(self, gammas, betas):
        """|amplitude|^2 of every bitstring, float64, on the host."""
        if self.backend == "numpy":
            return numpy_probabilities(self.n, self._host_arrays()[2], gammas, betas)
        from . import engine

        return engine.probabilities(self, gammas, betas, _PRECISION)

    def lr_probabilities(self, p, dg=DG, db=DB, u=None):
        return self.probabilities(*angles(p, dg, db, u))

    def energy(self, gammas, betas):
        """<sum w ZZ> with the original weights, the variational cost."""
        return float(self.w.sum()) - 2.0 * self.evaluate(gammas, betas)["expected_cut"]

    def describe(self):
        """Backend tag for the output files."""
        if self.backend == "numpy":
            return "numpy-fp64"
        from . import engine

        return (
            f"cudaq-{'nvidia-legacy' if self.n <= engine.LEGACY_MAX_N else 'nvidia'}-"
            f"{'fp64' if self.n <= engine.LEGACY_MAX_N else _PRECISION}"
        )


__all__ = [
    "Problem",
    "angles",
    "set_precision",
    "precision",
    "choose_backend",
    "hc_vector",
    "numpy_probabilities",
]
