"""closed_loop_validate is shared by two entrypoints, so every arg it reads must exist on both.

``train_predictor.py`` hands it a :class:`TrainConfig` dataclass, while
``train_grpo_predictor.py`` hands it the raw ``argparse.Namespace`` from its own
``get_args()``. A ``TrainConfig`` field added without the matching GRPO flag therefore
raises ``AttributeError`` at the first checkpoint-save epoch -- inside the
``if global_rank == 0:`` block, so rank 0 dies while the other ranks wait on the next
epoch's ``torch.distributed.barrier()`` until the NCCL timeout.

The source is read with ``ast`` rather than imported: ``train_grpo_predictor`` pulls in
torch/timm/wandb, which these tests do not need.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GRPO_PY = _REPO_ROOT / "diffusion_planner" / "train_grpo_predictor.py"
_TRAIN_PY = _REPO_ROOT / "diffusion_planner" / "diffusion_planner" / "train.py"


def _grpo_arg_dests() -> set[str]:
    """Every attribute ``train_grpo_predictor.get_args()`` puts on its Namespace."""
    tree = ast.parse(_GRPO_PY.read_text(encoding="utf-8"))
    get_args = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "get_args"
    )
    dests: set[str] = set()
    for node in ast.walk(get_args):
        is_add_argument = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        )
        if not is_add_argument:
            continue
        explicit = [kw.value for kw in node.keywords if kw.arg == "dest"]
        if explicit and isinstance(explicit[0], ast.Constant):
            dests.add(explicit[0].value)
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and str(arg.value).startswith("--"):
                dests.add(str(arg.value)[2:].replace("-", "_"))
                break
    return dests


def _args_attrs_read_by(func_name: str, path: Path) -> set[str]:
    """Attributes read as ``args.<name>`` inside ``func_name`` (nested functions included)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == func_name
    )
    return {
        node.attr
        for node in ast.walk(func)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "args"
    }


def _reads_sites_npz_root(test: ast.expr) -> bool:
    """Whether an ``if`` test gates on ``args.closed_loop_sites_npz_root``.

    The guard is currently ``and``-ed with ``is_final_save``; accept either shape so the
    test tracks the guard's meaning rather than its exact condition.
    """
    parts = test.values if isinstance(test, ast.BoolOp) else [test]
    return any(
        isinstance(part, ast.Attribute) and part.attr == "closed_loop_sites_npz_root"
        for part in parts
    )


def _sites_guard_source() -> str:
    """The verbatim ``if args.closed_loop_sites_npz_root ...:`` block of ``closed_loop_validate``.

    Located by content rather than line number so the test survives edits above it.
    """
    source = _TRAIN_PY.read_text(encoding="utf-8")
    tree = ast.parse(source)
    validate = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "closed_loop_validate"
    )
    guard = next(
        node
        for node in ast.walk(validate)
        if isinstance(node, ast.If) and _reads_sites_npz_root(node.test)
    )
    segment = ast.get_source_segment(source, guard)
    assert segment, "could not recover the source of the sites guard"
    return segment


def test_grpo_parser_defines_the_sites_flag():
    """Precondition: per-site closed-loop eval is a supported GRPO option, not a dead flag."""
    assert "closed_loop_sites_npz_root" in _grpo_arg_dests()


def test_grpo_parser_defines_every_arg_closed_loop_validate_reads():
    """Both callers of ``closed_loop_validate`` must satisfy every attribute it reads.

    Compares name sets rather than a fixed list, so a ``TrainConfig`` field added in future
    without the matching GRPO flag fails here instead of at someone's first save epoch.
    """
    missing = sorted(_args_attrs_read_by("closed_loop_validate", _TRAIN_PY) - _grpo_arg_dests())
    assert not missing, f"read by closed_loop_validate but absent from get_args: {missing}"


def test_sites_guard_runs_on_a_grpo_namespace(tmp_path: Path):
    """Run ``closed_loop_validate``'s site-discovery guard against a GRPO-shaped Namespace.

    The block's own source text is executed, with its collaborators stubbed, so the test
    tracks train.py rather than a paraphrase of it.
    """
    path_list = tmp_path / "path_list.json"
    path_list.write_text(json.dumps([str(tmp_path / "proj_a" / "site_1" / "manual")]))

    args = argparse.Namespace(**dict.fromkeys(_grpo_arg_dests(), ""))
    args.closed_loop_sites_npz_root = str(path_list)

    namespace = {
        "args": args,
        "json": json,
        "Path": Path,
        # Local of closed_loop_validate, not an args attribute; sites only run on the final save.
        "is_final_save": True,
        "discover_sites_with_vehicles_from_json": lambda *a, **k: {},
        "_object_mode_pairs": lambda modes: (("objects", False),),
        "run_labeled": lambda *a, **k: None,
    }
    try:
        exec(compile(_sites_guard_source(), "<train.py sites guard>", "exec"), namespace)
    except AttributeError as exc:
        pytest.fail(f"guard is not runnable on a GRPO Namespace: {exc}")
