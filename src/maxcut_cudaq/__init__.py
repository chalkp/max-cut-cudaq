"""LR-QAOA for MaxCut: closed forms (exact), instances (graphs), CUDA-Q circuits (qaoa)."""

import ctypes
import glob
import os
import sys


def _preload_nvrtc():
    # CuPy dlopen()s libnvrtc.so.13 to build its kernels. The CUDA-Q wheels ship it
    # under site-packages/nvidia/cu13/lib, which is not on the loader path, so pull
    # it in globally once instead of asking for LD_LIBRARY_PATH.
    for base in sys.path:
        for lib in sorted(
            glob.glob(os.path.join(base, "nvidia", "cu13", "lib", "libnvrtc.so.13*"))
        ):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                return
            except OSError:
                pass


_preload_nvrtc()
