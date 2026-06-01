"""Merge per-sample validation prediction errors into each npz's sidecar json.

valid_predictor.py writes one ``loss{i:08d}.json`` per sample (loss_ego_total,
loss_ego_3sec/5sec/8sec = FDE in metres, and various ego_* metrics). The mapping is
positional: valid_run.sh runs single-GPU with ``shuffle=False``, so the i-th loss file
corresponds to the i-th entry of the valid data list — the same convention
visualize_prediction.py relies on. The sidecar json sits next to the npz (``<npz>.json``,
written by the converter with is_skipped / skipping_info / pose).

This script copies each sample's loss dict into its sidecar under
``model_eval[<model_tag>]`` so scores live next to the frame. It is a re-runnable
post-step: re-running the converter regenerates the sidecar and drops these scores.

Usage:
    python3 annotate_npz_with_prediction_score.py \
        --valid_data_list /mnt/nvme/parse_rosbag_test/all_valid/valid_path_list.json \
        --predictions_dir <MODEL_DIR>/validation_result/predictions
"""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--valid_data_list", type=Path, required=True,
        help="The exact path_list.json passed to valid_run.sh (defines sample order).")
    parser.add_argument(
        "--predictions_dir", type=Path, required=True,
        help="<MODEL_DIR>/validation_result/predictions (holds loss{i}.json).")
    parser.add_argument(
        "--model_tag", type=str, default=None,
        help="Key under which scores are stored. Default: last 3 path parts of MODEL_DIR.")
    parser.add_argument("--dry_run", action="store_true", help="Check counts only; write nothing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = json.loads(args.valid_data_list.read_text())
    npz_list = data["files"] if isinstance(data, dict) else data
    loss_files = sorted(args.predictions_dir.glob("loss*.json"))

    if len(npz_list) != len(loss_files):
        raise SystemExit(
            f"count mismatch: {len(npz_list)} npz in list vs {len(loss_files)} loss files.\n"
            "The --valid_data_list must be exactly the one passed to valid_run.sh "
            "(single-GPU, shuffle=False) so the positional mapping holds.")

    model_tag = args.model_tag
    if model_tag is None:
        # SAVE_DIR layout: <MODEL_DIR>/validation_result/predictions
        model_dir = args.predictions_dir.parents[1]
        model_tag = "_".join(model_dir.parts[-3:])

    print(f"model_tag = {model_tag}")
    print(f"annotating {len(npz_list)} sidecar jsons"
          + (" (dry run)" if args.dry_run else "") + " ...")

    n_written = 0
    n_missing = 0
    for npz_path, loss_file in zip(npz_list, loss_files):
        sidecar = Path(npz_path).with_suffix(".json")
        if not sidecar.is_file():
            n_missing += 1
            continue
        scores = json.loads(loss_file.read_text())
        meta = json.loads(sidecar.read_text())
        meta.setdefault("model_eval", {})[model_tag] = scores
        if not args.dry_run:
            sidecar.write_text(json.dumps(meta, indent=2, sort_keys=True))
        n_written += 1

    print(f"done: {'would write' if args.dry_run else 'wrote'} {n_written}, "
          f"missing sidecar {n_missing}")


if __name__ == "__main__":
    main()
