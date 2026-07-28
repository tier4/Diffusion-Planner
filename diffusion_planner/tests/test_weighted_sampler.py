import json
import tempfile
import warnings
from pathlib import Path

import pytest
import torch
from diffusion_planner.utils.weighted_sampler import ClusterWeightedDistributedSampler


def _make_cluster_json(tmp_dir: str, data_list: list[str]) -> tuple[str, dict]:
    """Create a cluster JSON where first 2 paths are 'rare' and rest are 'common'."""
    clusters = {
        "cluster_id0": data_list[:2],
        "cluster_id1": data_list[2:],
    }
    path = str(Path(tmp_dir) / "clusters.json")
    with open(path, "w") as f:
        json.dump(clusters, f)
    return path, clusters


class TestWeightComputation:
    def test_rare_cluster_gets_higher_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(20)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42
            )

            rare_weight = sampler.weights[0].item()
            common_weight = sampler.weights[2].item()
            assert rare_weight > common_weight, (
                f"Rare cluster weight {rare_weight} should be > common {common_weight}"
            )

    def test_weights_length_matches_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(10)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42
            )
            assert len(sampler.weights) == len(data_list)


class TestIteration:
    def test_yields_correct_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(20)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42
            )
            indices = list(sampler)
            assert len(indices) == len(data_list)

    def test_indices_in_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(20)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42
            )
            indices = list(sampler)
            assert all(0 <= idx < len(data_list) for idx in indices)

    def test_rare_cluster_sampled_more(self):
        """Over many samples, rare cluster indices should appear more than their natural rate."""
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(100)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42
            )
            indices = list(sampler)
            rare_count = sum(1 for idx in indices if idx < 2)
            # Natural rate is 2/100 = 2%. With inverse-frequency weighting,
            # rare cluster should be sampled at ~50% (2 clusters, equal weight).
            assert rare_count > 10, (
                f"Rare cluster appeared {rare_count} times in 100 samples, expected >> 2"
            )


class TestDDP:
    def test_two_ranks_cover_full_dataset_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(20)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            s0 = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=2, rank=0, seed=42
            )
            s1 = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=2, rank=1, seed=42
            )
            i0, i1 = list(s0), list(s1)
            assert len(i0) == 10
            assert len(i1) == 10

    def test_two_ranks_get_different_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(20)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            s0 = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=2, rank=0, seed=42
            )
            s1 = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=2, rank=1, seed=42
            )
            i0, i1 = list(s0), list(s1)
            assert i0 != i1, "Different ranks should receive different index shards"

    def test_padding_with_non_divisible_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(21)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            s0 = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=2, rank=0, seed=42
            )
            s1 = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=2, rank=1, seed=42
            )
            assert len(list(s0)) == 11
            assert len(list(s1)) == 11

    def test_tiny_dataset_ddp_equal_shards(self):
        """Every rank gets equal shard length even when total_size < num_replicas."""
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(2)]
            clusters = {"cluster_id0": data_list[:1], "cluster_id1": data_list[1:]}
            cluster_path = str(Path(tmp) / "clusters.json")
            with open(cluster_path, "w") as f:
                json.dump(clusters, f)

            num_replicas = 8
            for rank in range(num_replicas):
                sampler = ClusterWeightedDistributedSampler(
                    data_list, cluster_path, num_replicas=num_replicas, rank=rank, seed=42
                )
                indices = list(sampler)
                assert len(indices) == sampler.num_samples, (
                    f"Rank {rank}: got {len(indices)} indices, expected {sampler.num_samples}"
                )

    def test_set_epoch_changes_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(50)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42
            )
            sampler.set_epoch(0)
            indices_e0 = list(sampler)
            sampler.set_epoch(1)
            indices_e1 = list(sampler)
            assert indices_e0 != indices_e1, "Different epochs should produce different orderings"


class TestEdgeCases:
    def test_multinomial_size_guard(self):
        """Datasets exceeding torch.multinomial's 2^24 category limit are rejected early."""
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(5)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            class BigList(list):
                def __len__(self):
                    return 2**24 + 1

            with pytest.raises(ValueError, match="2\\^24"):
                ClusterWeightedDistributedSampler(
                    BigList(data_list), cluster_path, num_replicas=1, rank=0
                )

    def test_missing_paths_get_neutral_weight(self):
        """Paths not in any cluster get the mean of matched weights (neutral rate).

        After normalization: rare > unmatched > common.
        """
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(10)]
            clusters = {
                "cluster_id0": data_list[:2],
                "cluster_id1": data_list[2:6],
            }
            cluster_path = str(Path(tmp) / "clusters.json")
            with open(cluster_path, "w") as f:
                json.dump(clusters, f)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                sampler = ClusterWeightedDistributedSampler(
                    data_list, cluster_path, num_replicas=1, rank=0, seed=42
                )
                matching_warnings = [x for x in w if "paths matched cluster JSON" in str(x.message)]
                assert len(matching_warnings) == 1

            assert len(sampler.weights) == 10
            assert sampler.matched_count == 6

            rare_weight = sampler.weights[0].item()
            common_weight = sampler.weights[3].item()
            unmatched_weight = sampler.weights[7].item()

            assert rare_weight > unmatched_weight > common_weight, (
                f"Expected rare ({rare_weight:.3f}) > unmatched ({unmatched_weight:.3f}) "
                f"> common ({common_weight:.3f})"
            )
            for i in range(6, 10):
                assert sampler.weights[i].item() == pytest.approx(unmatched_weight)

    def test_all_empty_clusters_raises_valueerror(self):
        """All-empty cluster JSON should raise ValueError, not ZeroDivisionError."""
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(5)]
            clusters = {"cluster_id0": [], "cluster_id1": []}
            cluster_path = str(Path(tmp) / "clusters.json")
            with open(cluster_path, "w") as f:
                json.dump(clusters, f)

            with pytest.raises(ValueError, match="Cluster JSON contains no paths"):
                ClusterWeightedDistributedSampler(
                    data_list, cluster_path, num_replicas=1, rank=0, seed=42
                )

    def test_no_matching_paths_raises_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(5)]
            clusters = {"cluster_id0": ["/other/path_0.npz", "/other/path_1.npz"]}
            cluster_path = str(Path(tmp) / "clusters.json")
            with open(cluster_path, "w") as f:
                json.dump(clusters, f)

            with pytest.raises(ValueError, match="No paths in data_list matched"):
                ClusterWeightedDistributedSampler(
                    data_list, cluster_path, num_replicas=1, rank=0, seed=42
                )

    def test_low_match_rate_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(10)]
            clusters = {"cluster_id0": data_list[:2]}
            cluster_path = str(Path(tmp) / "clusters.json")
            with open(cluster_path, "w") as f:
                json.dump(clusters, f)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                sampler = ClusterWeightedDistributedSampler(
                    data_list, cluster_path, num_replicas=1, rank=0, seed=42
                )
                matching_warnings = [x for x in w if "paths matched cluster JSON" in str(x.message)]
                assert len(matching_warnings) == 1
                assert sampler.matched_count == 2

    def test_cluster_counts_exposed(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(20)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42
            )
            assert sampler.cluster_counts == {"cluster_id0": 2, "cluster_id1": 18}
            assert sampler.matched_count == 20

    def test_cluster_counts_reflect_live_data_list(self):
        """cluster_counts should reflect the live data_list, not raw JSON totals."""
        with tempfile.TemporaryDirectory() as tmp:
            all_paths = [f"/data/sample_{i}.npz" for i in range(20)]
            clusters = {
                "cluster_id0": all_paths[:10],
                "cluster_id1": all_paths[10:],
            }
            cluster_path = str(Path(tmp) / "clusters.json")
            with open(cluster_path, "w") as f:
                json.dump(clusters, f)

            # data_list only includes 2 from cluster_id0 and 8 from cluster_id1
            data_list = all_paths[:2] + all_paths[10:18]

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42
            )
            assert sampler.cluster_counts == {"cluster_id0": 2, "cluster_id1": 8}
            assert sampler.matched_count == 10

    def test_prefix_mismatched_paths_still_match(self):
        """Cluster JSON with different path prefix still matches via canonicalization."""
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"data/train/20230101/1200/frame_{i}.npz" for i in range(10)]
            clusters = {
                "cluster_id0": [
                    f"/mnt/nfs/data/train/20230101/1200/frame_{i}.npz" for i in range(2)
                ],
                "cluster_id1": [
                    f"/mnt/nfs/data/train/20230101/1200/frame_{i}.npz" for i in range(2, 10)
                ],
            }
            cluster_path = str(Path(tmp) / "clusters.json")
            with open(cluster_path, "w") as f:
                json.dump(clusters, f)

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42
            )
            assert sampler.matched_count == 10, (
                f"Expected all 10 paths to match, got {sampler.matched_count}"
            )


class TestAlpha:
    def test_default_alpha_is_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(100)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            default = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42
            )
            explicit = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42, alpha=1.0
            )
            assert default.alpha == 1.0
            assert torch.equal(default.weights, explicit.weights)

    def test_alpha_zero_gives_uniform_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(100)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42, alpha=0.0
            )
            for w in sampler.weights.tolist():
                assert w == pytest.approx(1.0)

    def test_alpha_half_is_sqrt_of_full_ratio(self):
        """The rare/common weight ratio must go from R to R**alpha."""
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(100)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            full = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42, alpha=1.0
            )
            half = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42, alpha=0.5
            )
            # index 0 is in the rare cluster (2 samples), index 2 in the common one (98)
            ratio_full = full.weights[0].item() / full.weights[2].item()
            ratio_half = half.weights[0].item() / half.weights[2].item()

            assert ratio_full == pytest.approx(49.0, rel=1e-4)
            assert ratio_half == pytest.approx(ratio_full**0.5, rel=1e-4)

    def test_alpha_reduces_oversampling_multiplier(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(100)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            full = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42, alpha=1.0
            )
            soft = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42, alpha=0.25
            )
            assert soft.cluster_multipliers["cluster_id0"] < full.cluster_multipliers["cluster_id0"]
            assert soft.cluster_multipliers["cluster_id0"] > 1.0

    def test_negative_alpha_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(10)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            with pytest.raises(ValueError, match="alpha"):
                ClusterWeightedDistributedSampler(
                    data_list, cluster_path, num_replicas=1, rank=0, seed=42, alpha=-0.5
                )

    def test_nan_alpha_raises(self):
        """NaN must be rejected up front: it would silently make every weight NaN."""
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(10)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            with pytest.raises(ValueError, match="alpha"):
                ClusterWeightedDistributedSampler(
                    data_list,
                    cluster_path,
                    num_replicas=1,
                    rank=0,
                    seed=42,
                    alpha=float("nan"),
                )

    def test_inf_alpha_raises(self):
        """inf must be rejected at construction, not at the first __iter__.

        ``inf`` passes ``alpha >= 0``, but makes every weight NaN (inf/inf) and
        only fails later inside torch.multinomial -- after model init and dataset
        load, wasting a training launch.
        """
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(10)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            with pytest.raises(ValueError, match="alpha"):
                ClusterWeightedDistributedSampler(
                    data_list,
                    cluster_path,
                    num_replicas=1,
                    rank=0,
                    seed=42,
                    alpha=float("inf"),
                )

    def test_softened_alpha_with_unmatched_paths_keeps_ordering(self):
        """Regression guard on hunk ordering in ``_compute_weights``.

        The neutral-weight assignment for unmatched paths must run BEFORE the
        mean normalization, at *every* alpha -- not just the default. If someone
        moves normalization above the neutral block in the softened path, the
        unmatched weights stop landing between rare and common and this fails.
        """
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(10)]
            clusters = {
                "cluster_id0": data_list[:2],
                "cluster_id1": data_list[2:6],
            }
            cluster_path = str(Path(tmp) / "clusters.json")
            with open(cluster_path, "w") as f:
                json.dump(clusters, f)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sampler = ClusterWeightedDistributedSampler(
                    data_list, cluster_path, num_replicas=1, rank=0, seed=42, alpha=0.5
                )

            assert sampler.matched_count == 6
            rare_weight = sampler.weights[0].item()
            common_weight = sampler.weights[3].item()
            unmatched_weight = sampler.weights[7].item()

            assert rare_weight > unmatched_weight > common_weight, (
                f"alpha=0.5: expected rare ({rare_weight:.3f}) > "
                f"unmatched ({unmatched_weight:.3f}) > common ({common_weight:.3f})"
            )
            for i in range(6, 10):
                assert sampler.weights[i].item() == pytest.approx(unmatched_weight)

            # Assigning the neutral weight *before* normalization makes the overall
            # mean equal the matched mean, so the neutral weight normalizes to
            # exactly 1.0. Normalizing first leaves it at mean_matched / overall
            # mean != 1.0. This is the assertion that pins the hunk order.
            assert unmatched_weight == pytest.approx(1.0, rel=1e-9)
            assert sampler.weights.mean().item() == pytest.approx(1.0, rel=1e-9)

            # The softening must still be in effect (full inverse would give 2.0).
            assert rare_weight / common_weight == pytest.approx(2.0**0.5, rel=1e-4)

    def test_cluster_multipliers_match_normalized_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(100)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42
            )
            assert set(sampler.cluster_multipliers) == {"cluster_id0", "cluster_id1"}
            # index 0 -> cluster_id0, index 2 -> cluster_id1
            assert sampler.cluster_multipliers["cluster_id0"] == pytest.approx(
                sampler.weights[0].item()
            )
            assert sampler.cluster_multipliers["cluster_id1"] == pytest.approx(
                sampler.weights[2].item()
            )

    def test_multipliers_predict_observed_draw_counts(self):
        """A multiplier must be the *observed* oversampling factor, not just a number.

        Draws are accumulated over many epochs and compared against
        ``multiplier * cluster_size`` draws per epoch. This is what makes the
        multiplier meaningful for tuning alpha, and it fails if the recorded
        multipliers ever drift from the weights actually handed to
        ``torch.multinomial`` (e.g. if they were captured before normalization).
        """
        n_epochs = 100
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(100)]
            cluster_path, clusters = _make_cluster_json(tmp, data_list)

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42, alpha=0.5
            )
            index_to_cluster = {
                data_list.index(p): cid for cid, paths in clusters.items() for p in paths
            }

            observed = {cid: 0 for cid in clusters}
            for epoch in range(n_epochs):
                sampler.set_epoch(epoch)
                for idx in sampler:
                    observed[index_to_cluster[idx]] += 1

            total_draws = sum(observed.values())
            assert total_draws == n_epochs * len(data_list)

            for cid, count in sampler.cluster_counts.items():
                expected = sampler.cluster_multipliers[cid] * count * n_epochs
                assert observed[cid] == pytest.approx(expected, rel=0.08), (
                    f"{cid}: multiplier {sampler.cluster_multipliers[cid]:.3f} predicts "
                    f"{expected:.0f} draws over {n_epochs} epochs, observed {observed[cid]}"
                )


class TestShardConfigValidation:
    """A bad rank/num_replicas must crash, not silently hand out a short shard.

    ``indices[rank::num_replicas]`` yields fewer indices than ``__len__`` reports
    when ``rank >= num_replicas``. That rank then runs fewer optimizer steps than
    its peers and the whole job hangs on the next collective until the NCCL
    timeout -- hours into a multi-node run, with nothing in the logs explaining why.
    """

    def test_rank_at_or_above_num_replicas_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(10)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            with pytest.raises(ValueError, match="rank"):
                ClusterWeightedDistributedSampler(
                    data_list, cluster_path, num_replicas=4, rank=4, seed=42
                )

    def test_negative_rank_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(10)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            with pytest.raises(ValueError, match="rank"):
                ClusterWeightedDistributedSampler(
                    data_list, cluster_path, num_replicas=4, rank=-1, seed=42
                )

    def test_zero_num_replicas_raises(self):
        """Without the guard this is a bare ZeroDivisionError from math.ceil."""
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(10)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            with pytest.raises(ValueError, match="num_replicas"):
                ClusterWeightedDistributedSampler(
                    data_list, cluster_path, num_replicas=0, rank=0, seed=42
                )

    def test_shard_length_matches_len_for_every_valid_rank(self):
        """``__len__`` must not lie for any accepted rank -- DDP requires equal steps."""
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(103)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            for rank in range(4):
                sampler = ClusterWeightedDistributedSampler(
                    data_list, cluster_path, num_replicas=4, rank=rank, seed=42
                )
                assert len(list(sampler)) == len(sampler)


class TestAlphaUpperBound:
    """alpha > 1 inverts the weighting and silently collapses the epoch.

    Draws per cluster scale as ``matched ** alpha * n_c ** (1 - alpha)``, so above
    alpha=1 the largest cluster is starved. On an 18,000 + 10 split over 18,010
    draws: alpha=1.0 -> 8,935/9,075 (6,991 distinct), alpha=2.0 -> 7/18,003
    (17 distinct), alpha=5.0 -> 0/18,010 (10 distinct). Nothing raised before this
    guard; loss falls because the model memorizes.
    """

    @pytest.mark.parametrize("alpha", [1.0001, 2.0, 5.0, 200.0])
    def test_alpha_above_one_raises(self, alpha):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(1000)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            with pytest.raises(ValueError, match="alpha"):
                ClusterWeightedDistributedSampler(
                    data_list, cluster_path, num_replicas=1, rank=0, seed=42, alpha=alpha
                )

    @pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 1.0])
    def test_alpha_within_bounds_accepted(self, alpha):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(100)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42, alpha=alpha
            )
            assert sampler.alpha == alpha


class TestDuplicateClusterAssignment:
    def test_path_in_two_clusters_raises(self):
        """Last-write-wins would silently decide the cluster by JSON iteration order,
        while matched_count still reports a full match."""
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/train/loc/train/d/t/sample_{i}.npz" for i in range(10)]
            clusters = {
                "cluster_id0": data_list[:5],
                "cluster_id1": data_list[4:],  # sample_4 claimed twice
            }
            cluster_path = str(Path(tmp) / "clusters.json")
            with open(cluster_path, "w") as f:
                json.dump(clusters, f)

            with pytest.raises(ValueError, match="more than"):
                ClusterWeightedDistributedSampler(
                    data_list, cluster_path, num_replicas=1, rank=0, seed=42
                )

    def test_canonical_key_collision_across_roots_raises(self):
        """``data_path_to_rel`` anchors on the first train/valid component, so two
        NPZs under different dataset roots can collapse to one key."""
        with tempfile.TemporaryDirectory() as tmp:
            a = "/mnt/nvme1/dsA/loc/train/20240101/120000/0001.npz"
            b = "/mnt/rdma/dsB/loc/train/20240101/120000/0001.npz"
            clusters = {"cluster_id0": [a], "cluster_id1": [b]}
            cluster_path = str(Path(tmp) / "clusters.json")
            with open(cluster_path, "w") as f:
                json.dump(clusters, f)

            with pytest.raises(ValueError, match="more than"):
                ClusterWeightedDistributedSampler(
                    [a, b], cluster_path, num_replicas=1, rank=0, seed=42
                )

    def test_same_path_repeated_within_one_cluster_is_allowed(self):
        """A duplicate inside a single cluster is not ambiguous -- do not reject it."""
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(10)]
            clusters = {"cluster_id0": data_list[:2] + data_list[:2], "cluster_id1": data_list[2:]}
            cluster_path = str(Path(tmp) / "clusters.json")
            with open(cluster_path, "w") as f:
                json.dump(clusters, f)

            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42
            )
            assert sampler.matched_count == 10


class TestRanksPartitionOneStream:
    def test_interleaved_shards_reconstruct_single_rank_stream(self):
        """Every rank must draw the SAME global sequence and slice it by stride.

        ``__iter__`` seeds only on ``seed + epoch`` deliberately -- no rank term.
        Without this test, adding a ``+ rank`` to the seed still passes every other
        DDP test (shards still differ, lengths still match) while silently giving
        each rank its own overlapping sample stream.
        """
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(20)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)

            reference = list(
                ClusterWeightedDistributedSampler(
                    data_list, cluster_path, num_replicas=1, rank=0, seed=42
                )
            )
            shards = [
                list(
                    ClusterWeightedDistributedSampler(
                        data_list, cluster_path, num_replicas=2, rank=r, seed=42
                    )
                )
                for r in range(2)
            ]
            interleaved = [idx for pair in zip(*shards) for idx in pair]
            assert interleaved == reference


class TestRepeatFlags:
    def test_repeat_flags_none_when_not_tracking(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(20)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)
            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42
            )
            list(iter(sampler))
            assert sampler.repeat_flags is None
            assert sampler.repeat_flags_epoch is None

    def test_repeat_flags_length_matches_num_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(20)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)
            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42, track_repeats=True
            )
            list(iter(sampler))
            assert len(sampler.repeat_flags) == sampler.num_samples
            assert sampler.repeat_flags_epoch == 0

    def test_first_occurrence_false_later_true(self):
        """Flag i is True iff indices[i] appeared earlier in the same epoch."""
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(20)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)
            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42, track_repeats=True
            )
            indices = list(iter(sampler))
            seen = set()
            expected = []
            for idx in indices:
                expected.append(idx in seen)
                seen.add(idx)
            assert sampler.repeat_flags == expected

    def test_some_repeats_actually_occur(self):
        """Guard against a vacuous test: with 2 rare of 20, replacement must repeat."""
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(20)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)
            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42, track_repeats=True
            )
            list(iter(sampler))
            assert any(sampler.repeat_flags), "expected at least one repeat draw"

    def test_deterministic_for_same_seed_and_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(20)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)
            kwargs = dict(num_replicas=1, rank=0, seed=42, track_repeats=True)
            a = ClusterWeightedDistributedSampler(data_list, cluster_path, **kwargs)
            b = ClusterWeightedDistributedSampler(data_list, cluster_path, **kwargs)
            list(iter(a))
            list(iter(b))
            assert a.repeat_flags == b.repeat_flags

    def test_flags_change_across_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(20)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)
            sampler = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=1, rank=0, seed=42, track_repeats=True
            )
            list(iter(sampler))
            epoch0 = list(sampler.repeat_flags)
            sampler.set_epoch(1)
            list(iter(sampler))
            assert sampler.repeat_flags_epoch == 1
            assert sampler.repeat_flags != epoch0

    def test_dedup_is_global_across_ranks(self):
        """A sample drawn on two ranks yields one first occurrence, not two.

        Ordinals must be computed BEFORE sharding. Interleaving the two ranks'
        flags must reconstruct the global first-occurrence pattern.
        """
        with tempfile.TemporaryDirectory() as tmp:
            data_list = [f"/data/sample_{i}.npz" for i in range(20)]
            cluster_path, _ = _make_cluster_json(tmp, data_list)
            r0 = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=2, rank=0, seed=42, track_repeats=True
            )
            r1 = ClusterWeightedDistributedSampler(
                data_list, cluster_path, num_replicas=2, rank=1, seed=42, track_repeats=True
            )
            idx0, idx1 = list(iter(r0)), list(iter(r1))

            global_indices, global_flags = [], []
            for a, b in zip(idx0, idx1):
                global_indices.extend([a, b])
            for a, b in zip(r0.repeat_flags, r1.repeat_flags):
                global_flags.extend([a, b])

            seen = set()
            expected = []
            for idx in global_indices:
                expected.append(idx in seen)
                seen.add(idx)
            assert global_flags == expected
