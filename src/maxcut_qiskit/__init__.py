"""LR-QAOA for MaxCut: instances (graphs), Qiskit Aer circuits (qaoa)."""

import os


def worker_pool(jobs, max_tasks=None, executor=False):
    """worker_pool() - process pool for the experiment scripts
    @jobs: number of workers
    @max_tasks: recycle a worker after this many tasks
    @executor: return a ProcessPoolExecutor instead of a Pool

    Workers are spawned, not forked (CUDA does not survive a fork). BLAS and
    OpenMP threads, Aer's CPU path included, are capped at cores / jobs so the
    workers do not fight each other.
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
