#!/bin/bash
# Launch full evaluation scoring with CPU-pinned workers, then Stage 2 + Stage 3.
#
# Each worker gets 4 dedicated cores via taskset. Without this, 4 unpinned
# workers spawn ~247 threads on 16 cores (load avg 69) and thrash: measured
# throughput was 1.18 scenes/s, barely above a single worker's 1.08, while the
# GPU idled at 64%. Thread count does not affect results -- verified bit-for-bit
# identical Energy Scores at 4 threads vs unlimited.
#
# Uses the shard path lists already written by run_full_eval.py, so the shard
# split (and therefore --resume) matches the earlier partial run exactly.
#
# Usage: nohup bash human_match_prototype/launch_full_eval.sh > data/per_scene_eval/full_eval.log 2>&1 &

set -u
cd /home/chenglin/workspace/Diffusion-Planner-human-match

OUT=data/per_scene_eval/full_eval
MODEL=/home/chenglin/workspace/dp_exp_models/original_backup
MIRROR=data/per_scene_eval/mirror
TOTAL=156204

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

echo "=== Full Validation Evaluation (CPU-pinned) ==="
echo "Started: $(date)"

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

echo "[Step 1] Resuming from $(count_rows) already-scored scenes"

# Step 2: launch 4 pinned workers on the pre-existing shard path lists
echo "[Step 2] Starting 4 CPU-pinned GPU workers..."
CORES=("0-3" "4-7" "8-11" "12-15")
PIDS=()
for i in 0 1 2 3; do
    taskset -c "${CORES[$i]}" python3 -m human_match_prototype.score_scenes \
        --npz_list "$OUT/shard_${i}_paths.json" \
        --output "$OUT/shard_${i}.csv" \
        --model_dir "$MODEL" \
        --device cuda \
        --mirror "$MIRROR" \
        --num_samples 64 \
        --seed 0 \
        --temperature 1.0 \
        --resume \
        > "$OUT/worker_${i}.log" 2>&1 &
    PIDS+=($!)
    echo "  Worker $i: cores ${CORES[$i]}, PID ${PIDS[$i]}"
done

# Progress monitor
START=$(date +%s)
START_ROWS=$(count_rows)
while kill -0 "${PIDS[0]}" 2>/dev/null || kill -0 "${PIDS[1]}" 2>/dev/null \
   || kill -0 "${PIDS[2]}" 2>/dev/null || kill -0 "${PIDS[3]}" 2>/dev/null; do
    sleep 300
    NOW=$(date +%s)
    ROWS=$(count_rows)
    python3 - "$ROWS" "$START_ROWS" "$((NOW - START))" "$TOTAL" <<'PY'
import sys
rows, start_rows, elapsed, total = map(int, sys.argv[1:5])
done_now = rows - start_rows
rate = done_now / elapsed if elapsed else 0
eta = (total - rows) / rate / 3600 if rate > 0 else float("inf")
print(f"  Progress: {rows}/{total} ({rows/total*100:.1f}%) | {rate:.2f} scenes/s | ETA: {eta:.1f}h", flush=True)
PY
done

for i in 0 1 2 3; do
    wait "${PIDS[$i]}" && echo "  Worker $i finished OK" || echo "  Worker $i exited nonzero"
done
echo "[Step 2] Scoring complete: $(count_rows) rows"

# Step 3: merge shard CSVs
echo "[Step 3] Merging shard CSVs..."
python3 - "$OUT" <<'PY'
import sys
from pathlib import Path
from human_match_prototype.run_full_eval import merge_shard_csvs
out = Path(sys.argv[1])
shards = sorted(out.glob("shard_*.csv"))
n = merge_shard_csvs(shards, out / "scores_full.csv")
print(f"  Merged {n} rows from {len(shards)} shards")
PY

# Step 4: Stage 2 -- rank and select
echo "[Step 4] Running rank and select..."
python3 -m human_match_prototype.rank_and_select \
  --scores "$OUT/scores_full.csv" \
  --output_dir "$OUT/results"

# Step 5: Stage 3 -- render report
echo "[Step 5] Rendering report..."
python3 -m human_match_prototype.render_report \
  --scores "$OUT/results/ranked.csv" \
  --review_set "$OUT/results/review_set.csv" \
  --output "$OUT/results/report.html" \
  --model_dir "$MODEL" \
  --device cuda

echo "=== DONE ==="
echo "Finished: $(date)"
echo "Results: $OUT/results/"
