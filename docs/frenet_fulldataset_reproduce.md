# Reproducing the frenet full-dataset base training

This branch (`snapshot/frenet-fulldataset-s3407`) is a byte-exact snapshot of the
code that produced a full-dataset base model whose only difference from the
previous baseline recipe is the data augmentation: `quintic` -> `frenet`.
Everything else — base commit, dataset, hyper-parameters, seed, LR schedule,
EMA — was held fixed, so the run is a controlled single-variable comparison.

Dataset paths and per-run metrics are deliberately not recorded here; they live
with the campaign artifacts.

## Branch layout

| commit | what ran | epochs |
|---|---|---|
| `snapshot: ... (epochs 1-50)` | base commit + frenet augmenter + EPDMS wiring + DDP store fix | 1-50 |
| `snapshot: ... perf rewrite (epochs 51-80)` | 3-file augmenter/EPDMS speed rewrite, bit-identical outputs | 51-80 |
| `snapshot: ... evaluation opt-out` | eval path only, imported by neither training entrypoint | none |

Check out the branch head to reproduce the run end to end. The mid-run swap was
verified bit-identical (37 tensors, worst delta 0.000e+00) before adoption, so
the head reproduces epochs 1-50 as well; check out the first commit only if you
want the slower tree the first half literally executed on.

## Invocation

8x GPU on one node, `torch.distributed.run --nnodes 1 --nproc-per-node 8`:

```
train_predictor.py \
  --exp_name <name> --save_dir <out> \
  --train_set_list <train list> --valid_set_list <valid list> \
  --augment_type frenet \
  --use_data_augment True \
  --augment_prob 0.5 \
  --num_refine 20 \
  --ego_past_noise_std 0.1 \
  --use_smoothing_future_trajectory True \
  --diffusion_model_type x_start \
  --train_epochs 80 \
  --batch_size 512 \
  --learning_rate 1e-4 \
  --warm_up_epoch 5 \
  --seed 3407 \
  --save_utd 10 \
  --num_workers 8 \
  --use_ema True --ema_decay 0.999 \
  --deterministic True \
  --enable_epdms_eval True \
  --enable_temporal_stability_eval True \
  --enable_replan_consistency_eval True \
  --use_wandb False
```

`--closed_loop_npz_root` is deliberately absent: `closed_loop_validate` no-ops
when unset. The original run passed it until epoch 50, when the shared list it
pointed at was deleted and the job died with `FileNotFoundError`; it was dropped
on resume. It contributes nothing to the trained weights.

## Things that will bite you

- **Evaluate the EMA weights, never `ckpt["model"]`.** Both are stored. The raw
  iterates run at a constant LR that only anneals over the last ten epochs, and
  they carry large epoch-to-epoch jitter — measured on four consecutive
  checkpoints, EMA suppresses 94% of it laterally and 81% longitudinally.
  Extract with `torch.save({"model": ck["ema_state_dict"]}, tmp)`.
- **The ONNX written next to each checkpoint by the training loop is exported
  from RAW weights** (`use_ema=False` at both call sites in `train.py`). For a
  deployable artifact, re-export with `ros_scripts/torch2onnx.py --use_ema`.
- **`best_model/` is selected on validation lateral loss of the raw weights
  alone.** It ignores jerk, curvature and comfort, and it never sees the EMA
  copy. On this run it picked epoch 76, which ties the endpoint on lateral error
  but is materially rougher. Pick the checkpoint from a full EMA evaluation, not
  from that directory.
- **The frenet augmenter requires 3-column `neighbor_agents_future`** and
  `ego_shape` present in every NPZ; the pinned host code rotates a 4-column
  heading incorrectly. Assert this before launching, not after.
- **The LR schedule is misnamed.** `CosineAnnealingWarmUpRestarts` is a linear
  warmup followed by a multiplicative schedule with factor 1.0 — i.e. flat. The
  only annealing is a hard override late in training. Do not assume the name.
- **`ddp.py` used a hardcoded shared FileStore path.** Concurrent jobs on one
  node collided, and a crashed job left a stale store that poisoned every
  subsequent launch. Set `DP_DIST_INIT_FILE` per job and remove it on exit.

## Validating a rebuild

Two independent evaluations of the same epoch-50 weights, run as separate jobs
on different nodes, agreed to four decimal places on every metric. If a
re-training diverges by more than that, the difference is real and not noise.
