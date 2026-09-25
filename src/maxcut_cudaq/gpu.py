"""CuPy helpers. Bit q of a basis index is qubit q, same as CUDA-Q."""

import cupy as cp
import numpy as np

_CUT = cp.ElementwiseKernel(
    "raw int32 src, raw int32 tgt, raw float64 w, int32 m",
    "float64 c",
    """
    double s = 0.0;
    for (int e = 0; e < m; ++e)
        s += w[e] * (double)(((i >> src[e]) ^ (i >> tgt[e])) & 1);
    c = s;
    """,
    "maxcut_cut_values",
)


def cut_values(n, src, tgt, w):
    """Cut weight of every bitstring in [0, 2^n), float64 on the device."""
    out = cp.empty(1 << n, dtype=cp.float64)
    _CUT(
        cp.asarray(src, cp.int32),
        cp.asarray(tgt, cp.int32),
        cp.asarray(w, cp.float64),
        np.int32(len(w)),
        out,
    )
    return out


def state_view(state, n, fp64):
    """state_view() - CuPy array on top of a cudaq.State, no copy
    @state: the state; the returned array holds a reference to it
    @n: number of qubits
    @fp64: True for complex128 amplitudes
    """
    dt = cp.complex128 if fp64 else cp.complex64
    mem = cp.cuda.UnownedMemory(state.getTensor().data(), (1 << n) * cp.dtype(dt).itemsize, state)
    return cp.ndarray((1 << n,), dt, cp.cuda.MemoryPointer(mem, 0))
