#!/bin/bash
# Score the matched train / train_sft samples, then compare against validation.
#
# Scores the UNION of the two samples once (290,992 scenes) rather than each
# sample separately, since they share ~21k paths. Per-subset statistics are
# recovered afterwards by joining scored rows back to each sample list.
#
# Same thread-cap trick as launch_full_eval.sh: 4 workers x 4 cores. Without the
# caps, ~247 threads thrash 16 cores and the GPU idles at 64%.
#
# Usage: nohup bash human_match_prototype/launch_train_eval.sh > data/train_eval/run.log 2>&1 &

set -u
cd /home/chenglin/workspace/Diffusion-Planner-human-match

OUT=data/train_eval/scored
MIRROR=/mnt/nvme1/chenglin/train_eval_mirror
MODEL=/home/chenglin/workspace/dp_exp_models/original_backup
LIST=data/train_eval/sample90_union.json
WORKERS=4

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

mkdir -p "$OUT"
TOTAL=$(python3 -c "import json;print(len(json.load(open('$LIST'))))")
echo "=== Train vs SFT evaluation ==="
echo "Started: $(date)   scenes=$TOTAL"

# Wait for the fetch to finish if it is still running
while pgrep -f "fetch_parallel" > /dev/null 2>&1; do
    echo "  $(date +%H:%M:%S) waiting on fetch: $(find $MIRROR -name '*.npz' 2>/dev/null | wc -l)/$TOTAL"
    sleep 300
done
echo "[fetch] complete: $(find $MIRROR -name '*.npz' 2>/dev/null | wc -l) files"

# Shard the union list
python3 - "$LIST" "$OUT" "$WORKERS" <<'PY'
import json, sys
from pathlib import Path
from human_match_prototype.run_full_eval import split_into_shards
paths = json.load(open(sys.argv[1])); out = Path(sys.argv[2]); n = int(sys.argv[3])
for i, sh in enumerate(split_into_shards(paths, n)):
    json.dump(sh, open(out / f"shard_{i}_paths.json", "w"))
print(f"  sharded into {n} lists")
PY

count_rows() {
    python3 - "$OUT" <<'PY'
import sys
from pathlib import Path
t = 0
for s in Path(sys.argv[1]).glob("shard_*.csv"):
    with open(s) as f:
        t += max(0, sum(1 for _ in f) - 1)
print(t)
PY
}

echo "[score] launching $WORKERS pinned workers (resuming from $(count_rows))"
CORES=("0-3" "4-7" "8-11" "12-15")
PIDS=()
for i in 0 1 2 3; do
    taskset -c "${CORES[$i]}" python3 -m human_match_prototype.score_scenes \
        --npz_list "$OUT/shard_${i}_paths.json" \
        --output "$OUT/shard_${i}.csv" \
        --model_dir "$MODEL" --device cuda --mirror "$MIRROR" \
        --num_samples 64 --seed 0 --temperature 1.0 --resume \
        > "$OUT/worker_${i}.log" 2>&1 &
    PIDS+=($!)
done

START=$(date +%s); START_ROWS=$(count_rows)
while kill -0 "${PIDS[0]}" 2>/dev/null || kill -0 "${PIDS[1]}" 2>/dev/null \
   || kill -0 "${PIDS[2]}" 2>/dev/null || kill -0 "${PIDS[3]}" 2>/dev/null; do
    sleep 300
    python3 - "$(count_rows)" "$START_ROWS" "$(( $(date +%s) - START ))" "$TOTAL" <<'PY'
import sys
rows, s0, el, tot = map(int, sys.argv[1:5])
rate = (rows - s0) / el if el else 0
eta = (tot - rows) / rate / 3600 if rate > 0 else float("inf")
print(f"  Progress: {rows}/{tot} ({rows/tot*100:.1f}%) | {rate:.2f} scenes/s | ETA: {eta:.1f}h", flush=True)
PY
done
for i in 0 1 2 3; do wait "${PIDS[$i]}" || echo "  worker $i exited nonzero"; done

echo "[merge] combining shards"
python3 - "$OUT" <<'PY'
import sys
from pathlib import Path
from human_match_prototype.run_full_eval import merge_shard_csvs
out = Path(sys.argv[1])
n = merge_shard_csvs(sorted(out.glob("shard_*.csv")), out / "scores_union.csv")
print(f"  merged {n} rows")
PY

echo "[compare] train vs train_sft vs validation"
python3 -m human_match_prototype.compare_splits \
  --union_scores "$OUT/scores_union.csv" \
  --train_list data/train_eval/sample90_train.json \
  --sft_list data/train_eval/sample90_train_sft.json \
  --valid_scores data/per_scene_eval/full_eval/scores_full.csv \
  --output_dir data/train_eval/results \
  --match_n 90000

echo "=== DONE ==="
echo "Finished: $(date)"
