# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Registry and per-batch force-mask dispatch for augmentation on repeat draws."""

import torch

# Canonical application order. NOT user-configurable: "state_perturbation" must run
# last because it terminates with centric_transform(), which every other augmentation
# assumes has not yet happened. This order reproduces the sequence the individual
# augmentation PRs established in train_epoch.
AUG_ORDER: tuple[str, ...] = (
    "flip",
    "neighbor_dropout",
    "neighbor_noise",
    "turn_indicator",
    "traffic_light",
    "state_perturbation",
)

# Registry name -> the CLI flag that enables it, for error messages that tell the
# operator what to actually type.
_USE_FLAG: dict[str, str] = {
    "flip": "--use_flip_augment",
    "neighbor_dropout": "--use_neighbor_dropout",
    "neighbor_noise": "--use_neighbor_noise",
    "turn_indicator": "--use_turn_indicator_dropout",
    "traffic_light": "--use_traffic_light_dropout",
    "state_perturbation": "--use_data_augment",
}


def build_aug_pipeline(augmentations: dict) -> list[tuple[str, object]]:
    """Order enabled augmentations canonically.

    ``augmentations`` maps registry name -> augmentation object or None. Entries that
    are None are treated as disabled and omitted. Returns ``[(name, aug), ...]`` in
    AUG_ORDER.
    """
    unknown = sorted(set(augmentations) - set(AUG_ORDER))
    if unknown:
        raise ValueError(
            f"unknown augmentation name(s) {unknown}; valid names are {list(AUG_ORDER)}"
        )
    return [
        (name, augmentations[name]) for name in AUG_ORDER if augmentations.get(name) is not None
    ]


def resolve_repeat_aug_pool(
    pool_spec: str,
    pipeline: list[tuple[str, object]],
    augment_type: str,
) -> tuple[list[str], list[str]]:
    """Resolve --repeat_aug_pool into concrete registry names.

    Returns ``(pool_names, warnings)``. ``pool_names`` is in AUG_ORDER. Raises
    ValueError for anything that would make the run silently do the wrong thing --
    these are surfaced at launch, before the dataset scan.
    """
    enabled = [name for name, _ in pipeline]
    if not enabled:
        raise ValueError(
            "--force_aug_on_repeat was requested but no augmentations are enabled; "
            "enable at least one (e.g. --use_data_augment True)"
        )

    warnings: list[str] = []
    requested = [part.strip() for part in pool_spec.split(",")]
    requested = [part for part in requested if part]

    if requested:
        unknown = [name for name in requested if name not in AUG_ORDER]
        if unknown:
            raise ValueError(
                f"--repeat_aug_pool names unknown augmentation(s) {unknown}; "
                f"valid names are {list(AUG_ORDER)}"
            )
        not_enabled = [name for name in requested if name not in enabled]
        if not_enabled:
            flags = [_USE_FLAG[name] for name in not_enabled]
            raise ValueError(
                f"--repeat_aug_pool names {not_enabled}, which are not enabled. "
                f"Enable them with {flags} or drop them from the pool."
            )
        pool = [name for name in AUG_ORDER if name in set(requested)]
        if augment_type == "bridge" and "state_perturbation" in pool:
            warnings.append(
                "--repeat_aug_pool names state_perturbation while --augment_type bridge. "
                "Bridge forcing is NOT cost-neutral: its augment() searches per flagged row "
                "at roughly 912 ms/sample, so forcing adds proportional wall-time."
            )
    else:
        pool = list(enabled)
        if augment_type == "bridge" and "state_perturbation" in pool:
            # Excluded by default rather than silently doubling an already-prohibitive
            # cost. Nameable explicitly (above) for anyone who wants it anyway.
            pool = [name for name in pool if name != "state_perturbation"]
        if not pool:
            raise ValueError(
                "--repeat_aug_pool resolved to an empty pool. state_perturbation is "
                "excluded from the default pool under --augment_type bridge, and it is "
                "the only enabled augmentation. Either enable another augmentation, "
                "switch to --augment_type quintic, or name it explicitly with "
                "--repeat_aug_pool state_perturbation."
            )

    if pool == ["flip"]:
        warnings.append(
            "--repeat_aug_pool resolves to flip alone. Forced flip is deterministic, so "
            "two forced occurrences of the same sample mirror identically and remain "
            "duplicates of each other. Add another augmentation to the pool."
        )

    return pool, warnings


class ForcedAugmentationSelector:
    """Turns per-draw repeat flags into one force mask per augmentation name.

    For each repeat-flagged row exactly one pool member is chosen uniformly and forced
    on; the rest keep their own probability. Forcing the whole stack instead would mean
    rare clusters only ever see maximally-corrupted scenes.

    Flags are consumed with a running cursor advanced by each batch's ACTUAL row count.
    A computed ``j * batch_size`` offset would silently drift on any partial batch and
    misattribute every later mask.
    """

    def __init__(self, pool: list[str], seed: int = 0):
        if not pool:
            raise ValueError("pool must not be empty")
        unknown = sorted(set(pool) - set(AUG_ORDER))
        if unknown:
            raise ValueError(f"pool contains unknown augmentation(s) {unknown}")
        self.pool = list(pool)
        self.seed = seed
        self.forced_count = 0
        self.rows_consumed = 0
        # False until this epoch's flags are bound. train_epoch binds lazily on the
        # first batch and uses this to do it exactly once per epoch.
        self.is_bound = False
        self._flags: list[bool] | None = None
        self._generator: torch.Generator | None = None

    def start_epoch(
        self,
        epoch: int,
        repeat_flags: list[bool] | None,
        repeat_flags_epoch: int | None,
    ) -> None:
        """Bind this epoch's flags. Must be called after the loader's iterator exists."""
        self.is_bound = False
        if repeat_flags is None:
            raise ValueError(
                "sampler produced no repeat_flags; construct "
                "ClusterWeightedDistributedSampler with track_repeats=True"
            )
        # repeat_flags is regenerated inside __iter__, so reading it before iterating
        # returns the PREVIOUS epoch's flags. The stamp turns that silent misalignment
        # into a crash. It cannot catch set_epoch never being called at all -- both
        # values then stay equally stale -- which is covered by test coverage on
        # set_epoch instead.
        if repeat_flags_epoch != epoch:
            raise ValueError(
                f"stale repeat_flags: stamped for epoch {repeat_flags_epoch} but the "
                f"current epoch is {epoch}. Bind flags after the DataLoader's iterator "
                f"has been created."
            )
        self._flags = repeat_flags
        self.rows_consumed = 0
        self.forced_count = 0
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed + epoch)
        self.is_bound = True

    def masks_for_batch(self, batch_size: int, device) -> dict[str, torch.Tensor]:
        if self._flags is None or self._generator is None:
            raise RuntimeError("start_epoch must be called before masks_for_batch")

        end = self.rows_consumed + batch_size
        if end > len(self._flags):
            raise RuntimeError(
                f"reading past the end of repeat_flags: need rows "
                f"[{self.rows_consumed}, {end}) of {len(self._flags)}. The loader "
                f"delivered more rows than the sampler drew -- most likely a second "
                f"DataLoader iterator was created within one epoch."
            )
        flags = self._flags[self.rows_consumed : end]
        self.rows_consumed = end

        # Built on CPU because the flags originate as a Python list, then moved once.
        # Reading a mask back off the GPU would add a sync per batch.
        repeat = torch.tensor(flags, dtype=torch.bool)
        choice = torch.randint(len(self.pool), (batch_size,), generator=self._generator)
        self.forced_count += int(repeat.sum())

        return {
            name: (repeat & (choice == index)).to(device) for index, name in enumerate(self.pool)
        }


def validate_forced_aug_flags(args, pipeline: list[tuple[str, object]]) -> tuple[list, list]:
    """Validate forced-augmentation flags at launch, before the dataset scan.

    Returns ``(pool_names, warnings)``; an empty pool means the feature is off. All
    failures raise ValueError here rather than surfacing after model init and a
    150,000-file dataset load.
    """
    if not args.force_aug_on_repeat:
        return [], []
    if not args.cluster_json:
        raise ValueError(
            "--force_aug_on_repeat requires --cluster_json. Without the "
            "cluster-weighted sampler, draws are without replacement and no sample "
            "repeats within an epoch, so the flag would silently do nothing."
        )
    return resolve_repeat_aug_pool(args.repeat_aug_pool, pipeline, args.augment_type)


def duplicate_path_warning(data_list: list) -> str | None:
    """Warn if data_list repeats a path; dedup is by INDEX, so repeats evade it.

    Two entries for the same NPZ are distinct indices, so both are marked first
    occurrences and the identical scene trains twice unflagged. A warning rather than
    an error: a duplicated data_list is a dataset-generation problem that predates
    this feature and should not block a launch.
    """
    duplicates = len(data_list) - len(set(data_list))
    if duplicates == 0:
        return None
    return (
        f"data_list contains {duplicates} duplicate path(s). Repeat detection is by "
        f"dataset index, so duplicated paths are treated as distinct samples and the "
        f"same scene can train twice without being force-augmented."
    )
