"""PNG bytes must not depend on the number of render workers.

The neighbours are renamed to hex track UUIDs so the colour lookup takes its hashed branch, which
is the part that can differ between processes.
"""

import hashlib
from pathlib import Path

import numpy as np

from scenario_generation.render_pool import render_pool
from scenario_generation.reproducer_rollout import _draw_step

FIXTURE = Path(__file__).parent / "test_data" / "fixture_scene.npz"
N_FRAMES = 6


def _render(tmp_path, workers: int) -> list[str]:
    frame = dict(np.load(FIXTURE))
    np_dict = {k: v[None] for k, v in frame.items()}  # _draw_step indexes [0]
    uuids = [f"{0xA3F91C2E + i * 0x1111:08x}" for i in range(len(frame["neighbor_agents_past"]))]
    out_dir = tmp_path / f"workers{workers}"
    out_dir.mkdir()
    with render_pool(workers) as pool:
        pending = [
            pool.submit(
                _draw_step,
                np_dict,
                np.zeros((8, 4), dtype=np.float32),
                frame["ego_shape"],
                out_dir / f"{k:05d}.png",
                neighbor_ids=uuids,
                step=k,
                total=N_FRAMES,
            )
            for k in range(N_FRAMES)
        ]
        for f in pending:
            f.result()
    pngs = sorted(out_dir.glob("*.png"))
    assert len(pngs) == N_FRAMES, "nothing was drawn, so this proves nothing"
    return [hashlib.sha256(p.read_bytes()).hexdigest() for p in pngs]


def test_png_bytes_are_independent_of_worker_count(tmp_path):
    """3 workers over 6 frames, so a per-worker difference cannot hide in one process."""
    assert _render(tmp_path, 3) == _render(tmp_path, 1)
