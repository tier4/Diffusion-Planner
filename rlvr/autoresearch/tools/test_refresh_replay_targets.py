"""Unit tests for the monotone replay-memory refresh (max(frozen, fresh))."""

from __future__ import annotations

import json

import pytest

from rlvr.autoresearch.tools.refresh_replay_targets import build_rows, join


def _write(p, obj):
    p.write_text(json.dumps(obj))
    return p


def _rows(p, rows):
    with open(p, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


def test_build_rows_repairs_the_source_not_the_frozen_target(tmp_path):
    replay = _write(tmp_path / "replay.json", [str(tmp_path / "t1.npz")])
    prev = _rows(
        tmp_path / "prev.jsonl",
        [
            {
                "scene_path": str(tmp_path / "t1.npz"),
                "source_scene_path": str(tmp_path / "src1.npz"),
                "selected_total": -1.0,
                "repair_labels": ["expert_disagreement"],
            }
        ],
    )
    out = tmp_path / "rows.jsonl"
    stats = build_rows(replay, [prev], out)
    assert stats == {"replay": 1, "rows_written": 1, "missing": 0}
    row = json.loads(out.read_text().strip())
    # The fresh pass must re-repair the ORIGINAL window: the frozen target NPZ has
    # ego_agent_future overwritten, which would re-reference every deviation term.
    assert row["scene_path"] == str(tmp_path / "src1.npz")
    assert row["refresh_frozen_target"] == str(tmp_path / "t1.npz")
    assert row["refresh_frozen_total"] == -1.0
    assert row["repair_labels"] == ["expert_disagreement"]


def test_build_rows_fails_loudly_when_a_replay_scene_has_no_row(tmp_path):
    replay = _write(tmp_path / "replay.json", [str(tmp_path / "unknown.npz")])
    prev = _rows(tmp_path / "prev.jsonl", [])
    with pytest.raises(ValueError, match="no repaired row"):
        build_rows(replay, [prev], tmp_path / "rows.jsonl")
    stats = build_rows(replay, [prev], tmp_path / "rows.jsonl", allow_missing=True)
    assert stats["missing"] == 1 and stats["rows_written"] == 0


def _join_case(tmp_path, frozen_total, fresh_total, **kw):
    frozen, src, fresh_npz = (
        str(tmp_path / "frozen.npz"),
        str(tmp_path / "src.npz"),
        str(tmp_path / "fresh.npz"),
    )
    replay = _write(tmp_path / "replay.json", [frozen])
    prev = _rows(
        tmp_path / "prev.jsonl",
        [{"scene_path": frozen, "source_scene_path": src, "selected_total": frozen_total}],
    )
    fresh_rows = (
        []
        if fresh_total is None
        else [{"scene_path": fresh_npz, "source_scene_path": src, "selected_total": fresh_total}]
    )
    fr = _rows(tmp_path / "fresh.jsonl", fresh_rows)
    stats = join(replay, [prev], [fr], tmp_path / "out.json", tmp_path / "stats.json", **kw)
    return json.loads((tmp_path / "out.json").read_text()), stats, frozen, fresh_npz


def test_join_takes_the_fresh_target_when_it_scores_better(tmp_path):
    out, stats, _frozen, fresh = _join_case(tmp_path, -2.0, -1.0)
    assert out == [fresh]
    assert stats["improved_by_fresh"] == 1 and stats["kept_frozen"] == 0
    assert stats["mean_gain_on_improved"] == pytest.approx(1.0)


def test_join_keeps_the_frozen_target_when_the_policy_regressed(tmp_path):
    """The retention half: a drifted policy must not overwrite an old fix with a worse one."""
    out, stats, frozen, _fresh = _join_case(tmp_path, -1.0, -2.5)
    assert out == [frozen]
    assert stats["kept_frozen"] == 1 and stats["improved_by_fresh"] == 0


def test_join_keeps_frozen_when_the_policy_has_no_gate_passing_candidate(tmp_path):
    out, stats, frozen, _ = _join_case(tmp_path, -1.0, None)
    assert out == [frozen]
    assert stats["no_fresh_candidate"] == 1


def test_join_min_gain_adds_hysteresis_so_near_ties_stay_frozen(tmp_path):
    out, stats, frozen, _ = _join_case(tmp_path, -1.00, -0.99, min_gain=0.5)
    assert out == [frozen], "a 0.01 gain must not churn the memory when min_gain=0.5"
    assert stats["kept_frozen"] == 1


def test_join_is_monotone_target_score_never_decreases(tmp_path):
    """Whatever the fresh pass produces, the chosen target's score >= the frozen one's."""
    for frozen_total, fresh_total in ((-1.0, -3.0), (-1.0, 0.5), (-2.0, -2.0), (-2.0, None)):
        d = tmp_path / f"case_{frozen_total}_{fresh_total}"
        d.mkdir()
        out, _stats, frozen, fresh = _join_case(d, frozen_total, fresh_total)
        chosen = out[0]
        chosen_total = frozen_total if chosen == frozen else fresh_total
        assert chosen_total >= frozen_total


def test_rows_can_come_from_a_replay_memory_json(tmp_path):
    """A chain link only gets the previous round's MEMORY file, not its jsonl."""
    frozen, src = str(tmp_path / "t.npz"), str(tmp_path / "s.npz")
    replay = _write(tmp_path / "replay.json", [frozen])
    mem = _write(
        tmp_path / "memory.json",
        {
            "capacity": 10,
            "entries": [{"scene_path": frozen, "source_scene_path": src, "selected_total": -1.5}],
        },
    )
    out = tmp_path / "rows.jsonl"
    stats = build_rows(replay, [mem], out)
    assert stats["rows_written"] == 1
    row = json.loads(out.read_text().strip())
    assert row["scene_path"] == src and row["refresh_frozen_total"] == -1.5


def test_join_writes_a_map_of_only_the_improved_scenes(tmp_path):
    frozen, src, fresh_npz = (
        str(tmp_path / "f.npz"),
        str(tmp_path / "s.npz"),
        str(tmp_path / "n.npz"),
    )
    replay = _write(tmp_path / "replay.json", [frozen])
    prev = _rows(
        tmp_path / "prev.jsonl",
        [{"scene_path": frozen, "source_scene_path": src, "selected_total": -2.0}],
    )
    fr = _rows(
        tmp_path / "fresh.jsonl",
        [{"scene_path": fresh_npz, "source_scene_path": src, "selected_total": -1.0}],
    )
    join(
        replay,
        [prev],
        [fr],
        tmp_path / "o.json",
        tmp_path / "st.json",
        out_map=tmp_path / "map.json",
    )
    m = json.loads((tmp_path / "map.json").read_text())
    assert m[frozen]["new_path"] == fresh_npz and m[frozen]["new_total"] == -1.0


def test_join_map_excludes_scenes_where_frozen_won(tmp_path):
    frozen, src = str(tmp_path / "f.npz"), str(tmp_path / "s.npz")
    replay = _write(tmp_path / "replay.json", [frozen])
    prev = _rows(
        tmp_path / "prev.jsonl",
        [{"scene_path": frozen, "source_scene_path": src, "selected_total": -1.0}],
    )
    fr = _rows(
        tmp_path / "fresh.jsonl",
        [{"scene_path": str(tmp_path / "n.npz"), "source_scene_path": src, "selected_total": -3.0}],
    )
    join(
        replay,
        [prev],
        [fr],
        tmp_path / "o.json",
        tmp_path / "st.json",
        out_map=tmp_path / "map.json",
    )
    assert json.loads((tmp_path / "map.json").read_text()) == {}


def test_persist_into_memory_moves_path_AND_score_together(tmp_path):
    """The next round's join compares a fresh candidate against the STORED score, so a path
    moved without its score would make the following round decide keep/replace on stale data."""
    from rlvr.autoresearch.tools.refresh_replay_targets import persist_into_memory

    frozen, new = str(tmp_path / "f.npz"), str(tmp_path / "n.npz")
    mem = _write(
        tmp_path / "mem.json",
        {
            "capacity": 5,
            "entries": [
                {
                    "scene_path": frozen,
                    "source_scene_path": str(tmp_path / "s.npz"),
                    "selected_total": -2.0,
                },
                {"scene_path": str(tmp_path / "other.npz"), "selected_total": -9.0},
            ],
        },
    )
    mp = _write(
        tmp_path / "map.json", {frozen: {"new_path": new, "new_total": -0.5, "frozen_path": frozen}}
    )
    out = persist_into_memory(mem, mp)
    assert out == {"memory_entries": 2, "repointed_to_refreshed": 1}
    entries = json.loads(mem.read_text())["entries"]
    assert entries[0]["scene_path"] == new
    assert entries[0]["selected_total"] == -0.5, "score must move with the path"
    assert entries[0]["refreshed_from"] == frozen
    assert entries[1]["scene_path"].endswith("other.npz"), "untouched entries stay put"


def test_persist_is_idempotent_and_monotone_across_two_rounds(tmp_path):
    """Round N+1 must start from the refreshed target, and a second persist must not regress."""
    from rlvr.autoresearch.tools.refresh_replay_targets import persist_into_memory

    f, n1, n2 = (str(tmp_path / "f.npz"), str(tmp_path / "n1.npz"), str(tmp_path / "n2.npz"))
    mem = _write(
        tmp_path / "mem.json",
        {
            "entries": [
                {
                    "scene_path": f,
                    "source_scene_path": str(tmp_path / "s.npz"),
                    "selected_total": -3.0,
                }
            ]
        },
    )
    persist_into_memory(mem, _write(tmp_path / "m1.json", {f: {"new_path": n1, "new_total": -2.0}}))
    e = json.loads(mem.read_text())["entries"][0]
    assert (e["scene_path"], e["selected_total"]) == (n1, -2.0)
    # round N+1 improves it again, keyed off the NEW path
    persist_into_memory(
        mem, _write(tmp_path / "m2.json", {n1: {"new_path": n2, "new_total": -1.0}})
    )
    e = json.loads(mem.read_text())["entries"][0]
    assert (e["scene_path"], e["selected_total"]) == (n2, -1.0)
    assert e["selected_total"] > -3.0, "stored target score is monotone non-decreasing"


def test_join_raises_on_missing_frozen_score(tmp_path):
    """A replay row without selected_total is corrupted bookkeeping (repaired rows
    always carry it) — join must fail loudly, not silently retain the scene."""
    import json

    frozen = tmp_path / "frozen.npz"
    frozen.write_bytes(b"x")
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps([str(frozen)]))
    rows = tmp_path / "rows.jsonl"
    rows.write_text(
        json.dumps({"scene_path": str(frozen), "source_scene_path": "/tmp/src.npz"}) + "\n"
    )
    fresh = tmp_path / "fresh.jsonl"
    fresh.write_text(
        json.dumps(
            {
                "scene_path": str(tmp_path / "fresh.npz"),
                "source_scene_path": "/tmp/src.npz",
                "selected_total": 1.0,
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="selected_total"):
        join(
            replay_scenes=replay,
            prev_rows=[rows],
            fresh_rows=[fresh],
            out_list=tmp_path / "out.json",
            out_stats=tmp_path / "stats.json",
        )
