"""Versioned tar-shard dataset: packer, manifests, key-sets and a DDP-safe PyTorch loader.

Copied from the ``diffusion_planner.data_pipeline`` / ``diffusion_planner.utils.shard_*`` modules
of the tier4-main training tree so the new-architecture workspace can consume shards without a
package-name clash. Keep the two copies in sync until one becomes a shared dependency.
"""

FORMAT_VERSION = 1
PACKER_VERSION = "0.1.0"
