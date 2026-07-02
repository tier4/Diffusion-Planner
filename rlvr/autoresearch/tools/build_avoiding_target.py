#!/usr/bin/env python3
"""Build repaired curated targets for dangerous scenes.

Takes either:
  * a JSON list of NPZ paths (legacy static-collision mode), or
  * a JSONL of mined/classified scene rows with explicit labels.

For each scene, generate K guided candidates under the source model, classify
every candidate with the same dangerous-scene logic used in the mining step,
and keep only candidates that actually repair the mined issue while also
passing the global safety / lane / kinematic gates. The selected candidate is
written back into ``ego_agent_future`` for base SFT / IL training.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlvr.autoresearch.tools.classify_scene_failures import (
    _load_scene_thresholds,
    classify_loaded_scene_candidates_batch,
)
from rlvr.autoresearch.tools.eval_det_avoidance import load_model, load_npz_data
from rlvr.autoresearch.tools.reward_config_from_json import load_reward_config
from rlvr.grpo_trainer_batched import (
    _normalize_batch,
    _stack_scene_data,
    generate_all_scenes_batched,
)
from rlvr.reward import compute_reward_batch

_REPAIRABLE_LABELS = {"road_border_crossing", "static_collision", "moving_collision"}


def _parse_ego_shape(text: str) -> np.ndarray:
    vals = np.array([float(x) for x in text.split(",")], dtype=np.float32)
    if vals.shape != (3,):
        raise ValueError(f"--ego_shape must be WB,L,W, got {text!r}")
    return vals


def _output_name_for_scene(scene_path: str) -> str:
    path = Path(scene_path)
    parent = path.parent.name
    if not parent:
        return path.name
    return f"{parent}__{path.name}"


def _filtered_npz_payload(loaded) -> dict[str, Any]:
    return {
        key: value
        for key, value in loaded.items()
        if np.asarray(value).dtype.kind not in ("U", "S", "O")
    }


def _future4_to_3col(traj: np.ndarray) -> np.ndarray:
    if traj.ndim != 2 or traj.shape[1] != 4:
        raise ValueError(f"expected (T,4) future, got {traj.shape}")
    yaw = np.arctan2(traj[:, 3], traj[:, 2])
    return np.column_stack([traj[:, 0], traj[:, 1], yaw]).astype(np.float32)


def _load_rows(
    *,
    scene_rows_jsonl: Path | None,
    scenes_json: Path | None,
    allowed_labels: set[str] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if scene_rows_jsonl is not None:
        with open(scene_rows_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                labels = row.get("labels")
                if not labels:
                    label = row.get("label")
                    labels = [label] if label else []
                repair_labels = [str(label) for label in labels if str(label) in _REPAIRABLE_LABELS]
                if allowed_labels is not None:
                    repair_labels = [label for label in repair_labels if label in allowed_labels]
                if not repair_labels:
                    continue
                row = dict(row)
                row["repair_labels"] = repair_labels
                if "label" not in row:
                    row["label"] = repair_labels[0]
                rows.append(row)
        return rows

    if scenes_json is None:
        raise ValueError("one of --scene_rows_jsonl or --scenes is required")

    with open(scenes_json) as f:
        scene_paths = json.load(f)
    if not isinstance(scene_paths, list):
        raise ValueError(f"{scenes_json} must contain a JSON list of NPZ paths")
    for path in scene_paths:
        row = {
            "scene_path": str(path),
            "label": "static_collision",
            "labels": ["static_collision"],
            "repair_labels": ["static_collision"],
        }
        if allowed_labels is not None and "static_collision" not in allowed_labels:
            continue
        rows.append(row)
    return rows


def _passes_global_gates(label_row: dict[str, Any], reward_row) -> bool:
    return (
        reward_row.collision_step is None
        and not reward_row.rb_crossing
        and not reward_row.lane_crossing
        and not reward_row.static_crossing
        and not reward_row.kinematic_violated
        and label_row["moving_collision_step"] is None
    )


def _repairs_source_labels(
    source_labels: list[str],
    label_row: dict[str, Any],
    reward_row,
    *,
    min_static_margin: float,
    require_conflict_clear: bool,
) -> bool:
    if not _passes_global_gates(label_row, reward_row):
        return False
    if require_conflict_clear and bool(label_row.get("expert_disagreement", False)):
        return False
    for label in source_labels:
        if label == "road_border_crossing" and reward_row.rb_crossing:
            return False
        if label == "static_collision":
            if reward_row.static_crossing or float(reward_row.sc_min_dist) < min_static_margin:
                return False
        if label == "moving_collision":
            if (
                reward_row.collision_step is not None
                or label_row["moving_collision_step"] is not None
            ):
                return False
    return True


def _best_safe_candidate(
    source_row: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    reward_rows,
    *,
    min_static_margin: float,
    require_conflict_clear: bool,
) -> tuple[int | None, dict[str, Any]]:
    accepted: list[tuple[float, int]] = []
    for idx, (label_row, reward_row) in enumerate(zip(candidate_rows, reward_rows, strict=True)):
        if _repairs_source_labels(
            source_row["repair_labels"],
            label_row,
            reward_row,
            min_static_margin=min_static_margin,
            require_conflict_clear=require_conflict_clear,
        ):
            accepted.append((float(reward_row.total), idx))

    if not accepted:
        best_total = max(float(r.total) for r in reward_rows)
        best_sc = max(float(getattr(r, "sc_min_dist", -99.0)) for r in reward_rows)
        return None, {
            "reason": "no_safe_candidate",
            "best_total": best_total,
            "best_sc_min_dist": best_sc,
        }

    accepted.sort(reverse=True)
    _, idx = accepted[0]
    reward_row = reward_rows[idx]
    label_row = candidate_rows[idx]
    return idx, {
        "selected_total": float(reward_row.total),
        "selected_sc_min_dist": float(getattr(reward_row, "sc_min_dist", 99.0)),
        "selected_rb_min_dist": float(getattr(reward_row, "rb_min_dist", 99.0)),
        "selected_labels": list(label_row["labels"]),
        "selected_candidate_index": int(idx),
    }


@torch.no_grad()
def build_repaired_targets(
    *,
    model_path: str,
    rows: list[dict[str, Any]],
    reward_config_path: str,
    threshold_config_path: str,
    ego_shape: np.ndarray,
    out_dir: Path,
    out_rows_jsonl: Path | None,
    out_list: Path,
    min_static_margin: float,
    K: int,
    variant: str,
    gt_max_speed: float,
    scene_batch_size: int,
    noise_low: float,
    noise_high: float,
    device: torch.device,
    require_conflict_clear: bool,
    enable_conflict_detector: bool,
    use_route_cl_guidance: bool,
) -> tuple[list[str], list[dict[str, Any]]]:
    rcfg = load_reward_config(reward_config_path)
    thresholds = _load_scene_thresholds(threshold_config_path)
    model, model_args = load_model(model_path, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    repaired_rows: list[dict[str, Any]] = []
    unrepaired_rows: list[dict[str, Any]] = []

    class _Args:
        pass

    cls_args = _Args()
    cls_args.enable_conflict_detector = enable_conflict_detector
    cls_args.expert_disagreement_thresh = thresholds["expert_disagreement_thresh"]
    cls_args.expert_disagreement_sustain_steps = thresholds["expert_disagreement_sustain_steps"]

    for start in range(0, len(rows), scene_batch_size):
        batch_rows = rows[start : start + scene_batch_size]
        datas = []
        kept_rows = []
        for row in batch_rows:
            p = row["scene_path"]
            data = load_npz_data(p, device)
            npz_es = data["ego_shape"].detach().cpu().numpy().reshape(-1)[:3]
            if not np.allclose(npz_es, ego_shape, atol=1e-2):
                raise ValueError(
                    f"{p}: --ego_shape {ego_shape.tolist()} != NPZ ego_shape "
                    f"{npz_es.tolist()} (platform mismatch)"
                )
            datas.append(data)
            kept_rows.append(row)

        if not datas:
            continue

        norm_batch = _normalize_batch(_stack_scene_data(datas, device), model_args)
        trajs = generate_all_scenes_batched(
            model,
            model_args,
            norm_batch,
            K=K,
            noise_range=(noise_low, noise_high),
            device=device,
            gen_chunk_size=K,
            gt_max_speed=gt_max_speed,
            generation_variant=variant,
            use_route_cl_guidance=use_route_cl_guidance,
        )
        scene_paths = [str(row["scene_path"]) for row in kept_rows]
        candidate_rows_per_scene = classify_loaded_scene_candidates_batch(
            scene_paths,
            trajs,
            datas,
            rcfg,
            moving_near_thresh=thresholds["moving_near_thresh"],
            static_near_thresh=thresholds["static_near_thresh"],
            rb_near_thresh=thresholds["rb_near_thresh"],
            device=device,
            args=cls_args,
        )

        for row, data, scene_trajs, candidate_rows in zip(
            kept_rows, datas, trajs, candidate_rows_per_scene, strict=True
        ):
            reward_rows = compute_reward_batch(scene_trajs, data, rcfg)
            best_idx, meta = _best_safe_candidate(
                row,
                candidate_rows,
                reward_rows,
                min_static_margin=min_static_margin,
                require_conflict_clear=require_conflict_clear,
            )
            name = _output_name_for_scene(row["scene_path"])
            if best_idx is None:
                unrepaired_rows.append({**row, **meta})
                print(
                    f"  UNREPAIRED {name}: labels={','.join(row['repair_labels'])} "
                    f"best_total={meta['best_total']:+.1f}"
                )
                continue

            with np.load(row["scene_path"], allow_pickle=True) as loaded:
                raw = _filtered_npz_payload(loaded)
            raw["ego_agent_future"] = _future4_to_3col(
                scene_trajs[best_idx].detach().cpu().numpy().astype(np.float32)
            )
            out_path = out_dir / name
            np.savez(out_path, **raw)

            repaired = dict(row)
            repaired["source_scene_path"] = str(row["scene_path"])
            repaired["scene_path"] = str(out_path)
            repaired.update(meta)
            repaired_rows.append(repaired)
            print(
                f"  repaired {name}: labels={','.join(row['repair_labels'])} "
                f"slot={meta['selected_candidate_index']} total={meta['selected_total']:+.1f}"
            )

    if not repaired_rows:
        raise RuntimeError(
            "No repaired targets were produced; refusing to emit an empty training set"
        )

    out_list.parent.mkdir(parents=True, exist_ok=True)
    repaired_paths = [row["scene_path"] for row in repaired_rows]
    out_list.write_text(json.dumps(repaired_paths, indent=2))
    if out_rows_jsonl is not None:
        out_rows_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(out_rows_jsonl, "w") as f:
            for row in repaired_rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    if unrepaired_rows:
        unrepaired_path = out_list.with_name(out_list.stem + "_unrepaired.json")
        unrepaired_path.write_text(json.dumps(unrepaired_rows, indent=2))
        print(f"  unrepaired list -> {unrepaired_path}")

    print(f"repaired {len(repaired_rows)}/{len(rows)} -> {out_dir}")
    return repaired_paths, unrepaired_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="generation source model")
    ap.add_argument("--config", required=True, help="reward config JSON")
    ap.add_argument("--threshold_config", help="scene threshold config JSON")
    ap.add_argument("--ego_shape", required=True, help="WB,L,W — validated against each NPZ")
    ap.add_argument("--scene_rows_jsonl", help="JSONL with mined scene rows and labels")
    ap.add_argument("--scenes", help="JSON list of NPZs to repair (legacy static-collision mode)")
    ap.add_argument("--labels", help="comma-separated subset of labels to repair")
    ap.add_argument("--min_margin", type=float, required=True, help="required static clearance")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_list", required=True)
    ap.add_argument("--out_rows_jsonl", help="write repaired metadata rows")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--variant", default="rl_cl_soft_sweep_stretch")
    ap.add_argument("--gt_max_speed", type=float, default=9.0)
    ap.add_argument("--scene_batch_size", type=int, default=8)
    ap.add_argument("--noise_low", type=float, default=0.5)
    ap.add_argument("--noise_high", type=float, default=2.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow_conflict_candidates", action="store_true")
    ap.add_argument("--enable_conflict_detector", action="store_true")
    ap.add_argument("--disable_route_cl_guidance", action="store_true")
    args = ap.parse_args()

    if not args.scene_rows_jsonl and not args.scenes:
        raise ValueError("one of --scene_rows_jsonl or --scenes is required")
    if args.scene_rows_jsonl and not args.threshold_config:
        raise ValueError("--threshold_config is required with --scene_rows_jsonl")
    if args.scenes and not args.threshold_config:
        raise ValueError("--threshold_config is required")

    allowed_labels = None
    if args.labels:
        allowed_labels = {label.strip() for label in args.labels.split(",") if label.strip()}
        unknown = allowed_labels - _REPAIRABLE_LABELS
        if unknown:
            raise ValueError(f"unsupported repair labels: {sorted(unknown)}")

    rows = _load_rows(
        scene_rows_jsonl=Path(args.scene_rows_jsonl) if args.scene_rows_jsonl else None,
        scenes_json=Path(args.scenes) if args.scenes else None,
        allowed_labels=allowed_labels,
    )
    if not rows:
        raise RuntimeError("No repairable scene rows remained after label filtering")

    build_repaired_targets(
        model_path=args.model,
        rows=rows,
        reward_config_path=args.config,
        threshold_config_path=args.threshold_config,
        ego_shape=_parse_ego_shape(args.ego_shape),
        out_dir=Path(args.out_dir),
        out_rows_jsonl=Path(args.out_rows_jsonl) if args.out_rows_jsonl else None,
        out_list=Path(args.out_list),
        min_static_margin=float(args.min_margin),
        K=int(args.K),
        variant=str(args.variant),
        gt_max_speed=float(args.gt_max_speed),
        scene_batch_size=int(args.scene_batch_size),
        noise_low=float(args.noise_low),
        noise_high=float(args.noise_high),
        device=torch.device(args.device),
        require_conflict_clear=not bool(args.allow_conflict_candidates),
        enable_conflict_detector=bool(args.enable_conflict_detector),
        use_route_cl_guidance=not bool(args.disable_route_cl_guidance),
    )


if __name__ == "__main__":
    main()
