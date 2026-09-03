# tier4/Diffusion-Planner

This repository is a fork of <https://github.com/ZhengYinan-AIR/Diffusion-Planner>.

## Versioned shard dataset (data_pipeline)

Other teams: train from a dataset root + version tag — `--dataset_root <root> --dataset_version <tag>` (a tag is
byte-stable forever; `latest` moves when a new version is published) with a training/validation selection given as a
key-set file (`--train_key_set`) or a WHERE clause over the manifest (`--train_shard_filter "is_skipped IS NOT TRUE"`).
Samples are read through `diffusion_planner.data_pipeline.reader.ShardReader` or exported back to npz+json with
`python -m diffusion_planner.data_pipeline.pack_shards export …`. The per-sample data fields are exactly today's sidecar
fields; nothing was added. See `diffusion_planner/data_pipeline/validation/realdata_checklist.md`.
