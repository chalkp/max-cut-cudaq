"""CUDA-Q backend of qaoa.py: kernel, target selection and the on-device readout.

Only imported once a circuit actually runs on CUDA-Q, so numpy-only workers never
load cudaq or touch the GPU.
"""

import cudaq
import cupy as cp

from . import gpu

LEGACY_MAX_N = 17  # nvidia-legacy has less per-gate overhead up to here


@cudaq.kernel
def qaoa_circuit(
    n: int,
    src: list[int],
    tgt: list[int],
    w: list[float],
    deg: list[float],
    gammas: list[float],
    betas: list[float],
):
    q = cudaq.qvector(n)
    h(q)
    for k in range(len(gammas)):
        g = gammas[k]
        for e in range(len(src)):
            r1.ctrl(-4.0 * g * w[e], q[src[e]], q[tgt[e]])
        for i in range(n):
            r1(2.0 * g * deg[i], q[i])
        b = -2.0 * betas[k]
        for i in range(n):
            rx(b, q[i])


_TARGET = None


def select_target(n, precision):
    """Switch target only when it changes; returns (target, option)."""
    global _TARGET
    want = ("nvidia-legacy", "fp64") if n <= LEGACY_MAX_N else ("nvidia", precision)
    if want != _TARGET:
        cudaq.set_target(want[0], option=want[1])
        _TARGET = want
    return want


def state(P, gammas, betas, precision):
    """Run the circuit. Returns (cudaq.State, CuPy view); keep the view, it pins the state."""
    _, opt = select_target(P.n, precision)
    st = cudaq.get_state(
        qaoa_circuit,
        P.n,
        P.src.tolist(),
        P.tgt.tolist(),
        P.wn.tolist(),
        P.deg.tolist(),
        [float(x) for x in gammas],
        [float(x) for x in betas],
    )
    return st, gpu.state_view(st, P.n, opt == "fp64")


_PSTAR = cp.ReductionKernel(
    "T a, float64 c, float64 thr",
    "float64 y",
    "c >= thr ? (double)norm(a) : 0.0",
    "a + b",
    "y = a",
    "0",
    "maxcut_pstar",
)
_EXPECT = cp.ReductionKernel(
    "T a, float64 c", "float64 y", "(double)norm(a) * c", "a + b", "y = a", "0", "maxcut_expect"
)


def evaluate(P, gammas, betas, precision):
    """evaluate() - P(x*) and expected cut, reduced on the GPU
    @P: the Problem
    @gammas: cost angles, one per layer
    @betas: mixer angles, one per layer
    @precision: 'fp32' or 'fp64', ignored up to LEGACY_MAX_N qubits

    The state never leaves the device.

    Return: (P(x*), expected cut) as floats.
    """
    cut = P.cut_device()
    _, amp = state(P, gammas, betas, precision)
    return float(_PSTAR(amp, cut, P.threshold)), float(_EXPECT(amp, cut))


def probabilities(P, gammas, betas, precision):
    _, amp = state(P, gammas, betas, precision)
    return (cp.abs(amp) ** 2).astype(cp.float64).get()
