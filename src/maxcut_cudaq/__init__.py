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


def worker_pool(jobs, max_tasks=None, executor=False):
    """worker_pool() - process pool for the experiment scripts
    @jobs: number of workers
    @max_tasks: recycle a worker after this many tasks
    @executor: return a ProcessPoolExecutor instead of a Pool

    Workers are spawned, not forked (CUDA does not survive a fork). Long-lived
    CUDA-Q processes keep growing on the host, hence @max_tasks. BLAS threads
    are capped at cores / jobs so the eigensolvers do not fight each other.
    """
    import multiprocessing

    threads = str(max(1, (os.cpu_count() or 1) // max(1, jobs)))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = threads  # inherited by the spawned workers
    ctx = multiprocessing.get_context("spawn")
    if executor:
        from concurrent.futures import ProcessPoolExecutor

        return ProcessPoolExecutor(jobs, mp_context=ctx, max_tasks_per_child=max_tasks)
    return ctx.Pool(jobs, maxtasksperchild=max_tasks)
