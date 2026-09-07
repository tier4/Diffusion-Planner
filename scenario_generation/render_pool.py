"""The pool that renders per-step figures.

Matplotlib builds each figure in Python, so the GIL serialises a thread pool; worker processes
each hold their own interpreter and overlap for real.

Callers must be spawn-safe: an importable callable, picklable arguments, a
``if __name__ == "__main__":`` guard, and no mutation of what was submitted.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import Executor, ProcessPoolExecutor


def _pin_worker_threads() -> None:
    # torch sizes its intra-op pool from the machine's core count, so N workers each grab N_cores
    # threads and thrash.
    import torch

    torch.set_num_threads(1)


def render_pool(workers: int = 1) -> Executor:
    """A pool of ``workers`` renderers; anything below 1 is one worker.

    Spawn, not fork: the caller may hold a CUDA context. Workers start on the first submit, so a
    run that draws nothing spawns nothing. Each worker costs ~780 MB of RSS.
    """
    return ProcessPoolExecutor(
        max_workers=max(1, workers),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_pin_worker_threads,
    )
