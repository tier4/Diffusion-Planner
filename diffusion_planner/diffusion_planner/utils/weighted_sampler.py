import math
import warnings

import torch
from torch.utils.data import Sampler

from diffusion_planner.utils.path_key import data_path_to_rel
from diffusion_planner.utils.train_utils import openjson


class ClusterWeightedDistributedSampler(Sampler):
    """Inverse-frequency weighted sampler compatible with DDP.

    Reads a cluster assignment JSON (Fumiya's cluster.py output format) and assigns
    each sample a weight of ``(1 / cluster_frequency) ** alpha``. Rare clusters are
    sampled more often; no data is discarded.

    ``alpha`` controls how aggressive the reweighting is and must lie in ``[0, 1]``.
    At ``alpha=1.0`` (the default) every cluster receives an equal share of draws
    regardless of size. At ``alpha=0.0`` all weights are 1.0 and sampling is uniform
    -- note this is still sampling *with* replacement, so it is not identical to
    ``DistributedSampler``, which samples without replacement. Intermediate
    values interpolate: the multiplier ratio between any two clusters goes from
    ``R`` at alpha=1.0 to ``R ** alpha``. Values above 1.0 are rejected because they
    invert the weighting and starve the largest cluster.

    Weights are normalized to mean 1.0, so a sample's weight equals its
    oversampling multiplier. Per-cluster multipliers are exposed as
    ``cluster_multipliers`` for logging and tuning.

    Paths are canonicalized via ``data_path_to_rel`` before matching, so
    different prefixes (absolute vs relative, different mount points) are
    handled automatically.
    """

    def __init__(
        self,
        data_list: list[str],
        cluster_json_path: str,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        alpha: float = 1.0,
        track_repeats: bool = False,
    ):
        self.data_list = data_list
        if len(data_list) > 2**24:
            raise ValueError(
                f"data_list has {len(data_list)} entries, exceeding "
                f"torch.multinomial's 2^24 ({2**24:,}) category limit on CPU."
            )
        # Reject negatives *and* non-finite values (NaN, +/-inf). ``not alpha >= 0``
        # alone would let ``inf`` through: it makes every weight NaN (inf/inf), logs
        # ``nan x`` multipliers, and only dies at the first ``__iter__`` with an
        # opaque "invalid multinomial distribution" -- after model init and dataset
        # load, wasting a whole training launch.
        #
        # The upper bound matters just as much. Draws per cluster scale as
        # ``matched ** alpha * n_c ** (1 - alpha)``, so above alpha=1 the exponent on
        # ``n_c`` turns negative and the weighting INVERTS: the largest cluster is
        # starved. Measured on an 18,000 + 10 split over 18,010 draws/epoch --
        # alpha=1.0 gives 8,935/9,075 with 6,991 distinct samples, alpha=2.0 gives
        # 7/18,003 with 17 distinct, alpha=5.0 gives 0/18,010 with 10 distinct. That
        # trains on ten scenes while the loss falls, DDP stays healthy, and the only
        # hint is a ``0.00x`` in the startup log.
        if not (math.isfinite(alpha) and 0 <= alpha <= 1):
            raise ValueError(
                f"alpha={alpha} must be a finite number in [0, 1] "
                f"(1.0 = full inverse weighting, 0.0 = uniform). alpha > 1 over-inverts: "
                f"draws scale as n_c ** (1 - alpha), starving the LARGEST cluster and "
                f"collapsing the epoch onto a handful of samples."
            )
        # ``DistributedSampler`` validates these; a silently out-of-range rank is
        # worse here than a crash. ``indices[rank::num_replicas]`` yields a SHORT
        # shard when ``rank >= num_replicas`` while ``__len__`` still reports
        # ``ceil(N / num_replicas)``, so that rank runs fewer optimizer steps than
        # its peers and the job hangs on the next collective until the NCCL
        # timeout -- hours in, with no error explaining why.
        if num_replicas < 1:
            raise ValueError(f"num_replicas={num_replicas} must be >= 1")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank={rank} must be in [0, {num_replicas - 1}]")
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.alpha = alpha
        self.epoch = 0
        # Opt-in so a run that does not use forced augmentation pays nothing at all --
        # not even the once-per-epoch marking loop.
        self.track_repeats = track_repeats
        self.repeat_flags: list[bool] | None = None
        self.repeat_flags_epoch: int | None = None
        self.total_size = len(data_list)
        self.num_samples = math.ceil(self.total_size / self.num_replicas)

        (
            self.weights,
            self.cluster_counts,
            self.matched_count,
            self.cluster_multipliers,
        ) = self._compute_weights(cluster_json_path)

    def _compute_weights(
        self, cluster_json_path: str
    ) -> tuple[torch.Tensor, dict[str, int], int, dict[str, float]]:
        clusters = openjson(cluster_json_path)

        path_to_cluster: dict[str, str] = {}
        # A canonical key claimed twice would otherwise be silently last-write-wins,
        # and the offending sample lands in whichever cluster the JSON happens to
        # iterate last. Two ways it happens: the same path listed under two cluster
        # ids, or two distinct NPZs whose ``data_path_to_rel`` keys collide (that
        # helper anchors on the first "train"/"valid" component, so files under
        # different dataset roots sharing a location/date/time/frame layout collapse
        # to one key). Both cases make ``matched_count`` still read 100% while an
        # arbitrary subset carries the wrong cluster's weight -- unfixable from the
        # logs. cluster.py never emits duplicates, so this only fires on a
        # hand-edited or cross-machine JSON.
        collisions: dict[str, tuple[str, str]] = {}
        for cluster_id, paths in clusters.items():
            for p in paths:
                key = str(data_path_to_rel(p))
                if key in path_to_cluster and path_to_cluster[key] != cluster_id:
                    collisions[key] = (path_to_cluster[key], cluster_id)
                path_to_cluster[key] = cluster_id
        if collisions:
            shown = list(collisions.items())[:5]
            detail = "; ".join(f"{k} in both {a} and {b}" for k, (a, b) in shown)
            raise ValueError(
                f"Cluster JSON assigns {len(collisions)} canonical path(s) to more than "
                f"one cluster: {detail}"
                f"{' ...' if len(collisions) > len(shown) else ''}. "
                f"Assignments must be disjoint -- otherwise the applied cluster depends "
                f"on JSON iteration order and matched_count still reports a full match."
            )

        if not path_to_cluster:
            raise ValueError(
                f"Cluster JSON contains no paths "
                f"({len(clusters)} clusters, all empty). Check cluster JSON."
            )

        # First pass: identify cluster membership and count live matches
        sample_cluster = [None] * len(self.data_list)
        live_counts: dict[str, int] = {}
        matched = 0

        for i, path in enumerate(self.data_list):
            cid = path_to_cluster.get(str(data_path_to_rel(path)))
            if cid is not None:
                matched += 1
                sample_cluster[i] = cid
                live_counts[cid] = live_counts.get(cid, 0) + 1

        if matched == 0:
            raise ValueError(
                f"No paths in data_list matched cluster JSON "
                f"({len(path_to_cluster)} cluster entries). Check path formats."
            )

        # Compute frequencies from live counts
        cluster_freq = {cid: count / matched for cid, count in live_counts.items()}

        # Second pass: assign weights using live frequencies
        # ``alpha <= 1`` bounds every weight by ``matched``, which the 2^24 guard in
        # __init__ already caps -- so ``float.__pow__`` cannot overflow here.
        weights = torch.ones(len(self.data_list), dtype=torch.float64)
        for i, cid in enumerate(sample_cluster):
            if cid is not None:
                weights[i] = (1.0 / (cluster_freq[cid] + 1e-8)) ** self.alpha

        if matched < len(self.data_list):
            matched_mask = torch.tensor([c is not None for c in sample_cluster], dtype=torch.bool)
            mean_matched = weights[matched_mask].mean().item()
            weights[~matched_mask] = mean_matched
            warnings.warn(
                f"{matched}/{len(self.data_list)} paths matched cluster JSON. "
                f"Unmatched paths assigned neutral weight."
            )

        weights = weights / weights.mean()

        # Weights are mean-normalized, so a sample's weight is its oversampling
        # multiplier. Record one representative per cluster.
        multipliers: dict[str, float] = {}
        for i, cid in enumerate(sample_cluster):
            if cid is not None and cid not in multipliers:
                multipliers[cid] = weights[i].item()

        return weights, live_counts, matched, multipliers

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        padded_length = self.num_samples * self.num_replicas
        draws = torch.multinomial(
            self.weights, padded_length, replacement=True, generator=g
        ).tolist()

        if self.track_repeats:
            # Ordinals are computed on the GLOBAL draw list, BEFORE sharding, so a
            # sample drawn on rank 0 and rank 3 in the same step yields one first
            # occurrence and one repeat -- not two first occurrences. This needs no
            # collective: every rank seeds an identical generator with seed + epoch and
            # holds identical weights, so every rank computes the same `draws` and
            # differs only in which stride it slices.
            seen: set[int] = set()
            repeat: list[bool] = []
            for idx in draws:
                repeat.append(idx in seen)
                seen.add(idx)
            self.repeat_flags = repeat[self.rank : padded_length : self.num_replicas]
            self.repeat_flags_epoch = self.epoch

        indices = draws[self.rank : padded_length : self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
