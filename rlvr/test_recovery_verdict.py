"""The recovery verdict and the clip frame directory, pinned against two review findings.

1. A rollout that ends OFF the route must not be classified as recovered. The old
   ``lost`` test used ``near[-10:].any()``: one near-route sample in the final
   window was enough to escape it, and the settle mean -- taken over that single
   sample -- could then read as recovered.
2. Re-rendering the same clip identity must not encode frames left over from an
   earlier, longer run: ffmpeg is given ``*.png``.
"""

import numpy as np

from rlvr.autoresearch.tools.eval_recovery_route import summarize_rollout
from rlvr.autoresearch.tools.render_recovery_clip import fresh_frame_dir


def test_off_route_ending_is_lost_not_recovered():
    # 71 near-route steps, then 9 off-route; the one near sample in the final
    # window sits perfectly on the centreline. This is the exact trace from the
    # review: it used to return lost=False, recovered=True.
    route_dist = np.concatenate([np.zeros(71), np.full(9, 20.0)])
    usage = np.zeros(80)
    v = summarize_rollout(usage, route_dist)
    assert v["lost"] is True
    assert v["recovered"] is False


def test_near_final_window_with_low_usage_is_recovered():
    route_dist = np.zeros(80)
    usage = np.concatenate([np.linspace(1.0, 0.0, 70), np.full(10, 0.1)])
    v = summarize_rollout(usage, route_dist)
    assert v["lost"] is False
    assert v["recovered"] is True
    assert v["unsettled"] is False
    assert abs(v["usage_settle"] - 0.1) < 1e-9


def test_swerve_in_final_window_but_on_route_at_end_is_unsettled():
    # off-route three steps from the end, back inside at the final step: it did
    # not END off the route, so it is not lost -- but it did not settle either
    route_dist = np.zeros(80)
    route_dist[-3] = 20.0
    v = summarize_rollout(np.zeros(80), route_dist)
    assert v["lost"] is False
    assert v["recovered"] is False
    assert v["unsettled"] is True
    assert v["usage_settle"] is None


def test_off_route_steps_cannot_be_filtered_out_of_the_settle_mean():
    # nine near steps at usage 0.9 and one off-route step: the old code dropped
    # the off step and averaged the rest; now the window is not settled at all
    route_dist = np.zeros(80)
    route_dist[-5] = 20.0
    usage = np.full(80, 0.9)
    v = summarize_rollout(usage, route_dist)
    assert v["recovered"] is False and v["lost"] is False


def test_fresh_frame_dir_removes_stale_frames(tmp_path):
    png_dir = tmp_path / "clip_A"
    png_dir.mkdir()
    # a previous 80-frame run
    for i in range(80):
        (png_dir / f"{i:05d}.png").write_bytes(b"old")
    out = fresh_frame_dir(png_dir)
    assert out == png_dir
    assert list(png_dir.glob("*.png")) == []
    # a shorter rerender now writes only its own frames, and only those reach *.png
    for i in range(50):
        (png_dir / f"{i:05d}.png").write_bytes(b"new")
    frames = sorted(png_dir.glob("*.png"))
    assert len(frames) == 50
    assert all(f.read_bytes() == b"new" for f in frames)


def test_fresh_frame_dir_creates_missing_dir(tmp_path):
    png_dir = tmp_path / "nested" / "clip_B"
    assert fresh_frame_dir(png_dir).is_dir()
