r"""LR-QAOA (or any angle schedule) for MaxCut, scored exactly.

    |psi> = prod_k e^{+i beta_k sum X} e^{-i gamma_k H_C} |+>^n
    H_C   = sum_ij (w_ij / max|w|) Z_i Z_j

The mixer sign follows the LR-QAOA paper (their R_X(-2 beta)). One layer is |E|
rzz(2 gamma wn) and n rx(-2 beta); rzz(2t) is exactly e^{-i t ZZ}, no extra phase.

Small instances run on numpy, the rest on Aer (MAXCUT_BACKEND=auto|aer|numpy).
Nothing is sampled, and P(x*) counts every optimal bitstring.
"""

import os

import numpy as np

from .cpu import cut_values
from .cpu import probabilities as numpy_probabilities

DG, DB = 0.6, 0.3
NUMPY_NS_PER_AMP = 4.0  # one core, per amplitude per qubit per layer
AER_US_PER_GATE = 16.0  # Aer's circuit intake, more than a 2^13 state costs to update

_PRECISION = os.environ.get("MAXCUT_PRECISION", "fp64")


def set_precision(precision):
    """'fp64' (default) or 'fp32' for Aer circuits from now on; numpy stays fp64."""
    global _PRECISION
    _PRECISION = precision


def angles(p, dg=DG, db=DB, u=None):
    """Linear ramp, or schedule u[0..p]: gamma_k = u[k+1] dg, beta_k = (1 - u[k]) db."""
    u = np.linspace(0.0, 1.0, p + 1) if u is None else np.asarray(u, dtype=float)
    return u[1:] * dg, (1 - u[:-1]) * db


def choose_backend(n, n_edges):
    numpy_cost = NUMPY_NS_PER_AMP * 1e-3 * 2**n * n  # us per layer
    aer_cost = AER_US_PER_GATE * (n_edges + n)
    return "numpy" if numpy_cost < aer_cost else "aer"


class Problem:
    """One MaxCut instance.

    Cuts are scored with the original weights @w; the circuit sees w / max|w|.
    """

    def __init__(self, src, tgt, w, n=None, rtol=1e-9, backend=None):
        self.src = np.asarray(src, dtype=np.int64)
        self.tgt = np.asarray(tgt, dtype=np.int64)
        self.w = np.asarray(w, dtype=float)
        self.n = int(n if n is not None else max(self.src.max(), self.tgt.max()) + 1)
        self.scale = float(np.abs(self.w).max()) or 1.0
        self.wn = self.w / self.scale
        self.rtol = rtol
        backend = backend or os.environ.get("MAXCUT_BACKEND", "auto")
        self.backend = choose_backend(self.n, len(self.w)) if backend == "auto" else backend
        self._cut = self._hc = self._opt = None
        self.best = self.threshold = None

    def cuts(self):
        """Cut weight of every bitstring, indexed by int(bitstring, 2)."""
        if self._cut is None:
            self._cut = cut_values(self.n, self.src, self.tgt, self.w)
            self.best = float(self._cut.max())
            self.threshold = self.best - self.rtol * max(1.0, abs(self.best))
        return self._cut

    def optimum(self):
        self.cuts()
        return self.best

    def optimal_indices(self):
        """Every bitstring within @rtol of the best cut."""
        if self._opt is None:
            self._opt = np.flatnonzero(self.cuts() >= self.threshold)
        return self._opt

    def optimal_mask_host(self):
        return self.cuts() >= self.threshold

    def hc_host(self):
        """H_C per bitstring, normalized weights: sum wn - 2 cut / max|w|."""
        if self._hc is None:
            self._hc = float(self.wn.sum()) - 2.0 / self.scale * self.cuts()
        return self._hc

    def evaluate(self, gammas, betas):
        """evaluate() - run one circuit and score it
        @gammas: cost angles, one per layer
        @betas: mixer angles, one per layer

        Return: dict with success_probability, mean_ratio and expected_cut.
        """
        if self.backend == "numpy":
            pr = numpy_probabilities(self.n, self.hc_host(), gammas, betas)
            ps, ec = float(pr[self.optimal_indices()].sum()), float(pr @ self.cuts())
        else:
            from . import aer

            ps, ec = aer.evaluate(self, gammas, betas, _PRECISION)
        return dict(
            success_probability=ps,
            mean_ratio=ec / self.best if self.best else float("nan"),
            expected_cut=ec,
        )

    def lr(self, p, dg=DG, db=DB, u=None):
        return self.evaluate(*angles(p, dg, db, u))

    def probabilities(self, gammas, betas):
        """|amplitude|^2 of every bitstring, float64."""
        if self.backend == "numpy":
            return numpy_probabilities(self.n, self.hc_host(), gammas, betas)
        from . import aer

        return aer.probabilities(self, gammas, betas, _PRECISION)

    def energy(self, gammas, betas):
        """<sum w ZZ> with the original weights."""
        return float(self.w.sum()) - 2.0 * self.evaluate(gammas, betas)["expected_cut"]

    def describe(self):
        """Backend tag for the output files."""
        if self.backend == "numpy":
            return "numpy-fp64"
        from . import aer

        return f"aer-{aer.device()}-{_PRECISION}"
