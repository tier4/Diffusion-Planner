# §7 real-data validation checklist (mechanism only — nothing in production is touched)

Constraints: source tree is READ-ONLY; destination is a scratch directory on LOCAL disk (not RDMA);
the slice is defined by an explicit path list reviewed by a human; no npz is deleted anywhere.

1. Pick the slice: write `slice_paths.json` (JSON list of absolute npz paths, a few thousand frames from ONE
   partition of the chosen rule). Review it.
2. Inspect:   python -m diffusion_planner.data_pipeline.pack_shards inspect --source <tree> --partition-depth <N> --include '<glob>'
3. Pack:      python -m diffusion_planner.data_pipeline.pack_shards pack --source <tree> --dest /scratch/dp_test --base none --tag v1 \
                 --partition-depth <N> --path-list slice_paths.json
   Record: kept/rejected counts, shard count, wall time, manifest size (also run once with --with-neighbor-ids and compare sizes).
4. Bit-identity: python -m diffusion_planner.data_pipeline.pack_shards scrub --dest /scratch/dp_test --tag v1
   plus: for 100 random keys, `ShardReader.get(key)` vs `np.load(<source npz>)` via `encoding.arrays_bitexact`.
5. Uncompressed legacy slice (20260331-style tree): repeat 2–4 on a small slice; report size ratio.
6. Key-set:   pack_shards keyset --dest /scratch/dp_test --tag v1 --where "is_skipped IS NOT TRUE" --out ks.parquet
7. Mixing:    python -m diffusion_planner.data_pipeline.validation.mixing_test --dataset-root /scratch/dp_test --keyset ks.parquet --world-size 8 --workers 8 --batch-size 64
8. Throughput (on the training box, when GPUs are idle): validation.throughput_bench with --C 1,2,4,8 and --legacy-path-list slice_paths.json
9. Parity training (short): train.py --dataset_root /scratch/dp_test --dataset_version v1 --train_shard_filter "is_skipped IS NOT TRUE" \
   --valid_shard_filter "is_skipped IS NOT TRUE" … vs the npz path on slice_paths.json; compare loss curves.
10. Recipes:  run the equivalence test module against the slice (pytest -k recipes with DP_REAL_SLICE=... if wired) or the same
    steps by hand: legacy script output keys == adapter keys (ordered multiset).
11. Lifecycle: python -m diffusion_planner.data_pipeline.validation.lifecycle_rehearsal --source <slice dir> --dest /scratch/dp_lifecycle --partition-depth <N>

Report all numbers in the PR description; defaults (C, B, chunk, seek threshold) are set from steps 7–8.
