from types import SimpleNamespace

import pytest
import torch
from diffusion_planner.utils.forced_augmentation import (
    AUG_ORDER,
    ForcedAugmentationSelector,
    build_aug_pipeline,
    duplicate_path_warning,
    resolve_repeat_aug_pool,
    validate_forced_aug_flags,
)


class _FakeAug:
    """Stand-in for an augmentation object; identity is all these tests need."""

    def __init__(self, name):
        self.name = name


class TestAugOrder:
    def test_state_perturbation_is_last(self):
        """It terminates with centric_transform, which the others assume has not run."""
        assert AUG_ORDER[-1] == "state_perturbation"

    def test_order_matches_todays_train_epoch_sequence(self):
        assert AUG_ORDER == (
            "flip",
            "neighbor_dropout",
            "neighbor_noise",
            "turn_indicator",
            "traffic_light",
            "state_perturbation",
        )


class TestBuildAugPipeline:
    def test_orders_canonically_regardless_of_dict_order(self):
        augs = {
            "state_perturbation": _FakeAug("sp"),
            "flip": _FakeAug("flip"),
            "neighbor_noise": _FakeAug("nn"),
        }
        pipeline = build_aug_pipeline(augs)
        assert [name for name, _ in pipeline] == ["flip", "neighbor_noise", "state_perturbation"]

    def test_skips_none_entries(self):
        augs = {"flip": None, "state_perturbation": _FakeAug("sp")}
        pipeline = build_aug_pipeline(augs)
        assert [name for name, _ in pipeline] == ["state_perturbation"]

    def test_empty_dict_yields_empty_pipeline(self):
        assert build_aug_pipeline({}) == []

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="unknown augmentation"):
            build_aug_pipeline({"route_input": _FakeAug("r")})


class TestResolveRepeatAugPool:
    def _pipeline(self, *names):
        return [(n, _FakeAug(n)) for n in AUG_ORDER if n in names]

    def test_empty_spec_defaults_to_all_enabled(self):
        pipeline = self._pipeline("flip", "neighbor_noise")
        pool, warnings = resolve_repeat_aug_pool("", pipeline, augment_type="quintic")
        assert pool == ["flip", "neighbor_noise"]
        assert warnings == []

    def test_explicit_spec_restricts_pool(self):
        pipeline = self._pipeline("flip", "neighbor_noise", "state_perturbation")
        pool, _ = resolve_repeat_aug_pool("neighbor_noise", pipeline, augment_type="quintic")
        assert pool == ["neighbor_noise"]

    def test_explicit_spec_is_reordered_canonically(self):
        pipeline = self._pipeline("flip", "neighbor_noise")
        pool, _ = resolve_repeat_aug_pool("neighbor_noise,flip", pipeline, augment_type="quintic")
        assert pool == ["flip", "neighbor_noise"]

    def test_whitespace_and_empty_entries_tolerated(self):
        pipeline = self._pipeline("flip", "neighbor_noise")
        pool, _ = resolve_repeat_aug_pool(" flip , ,neighbor_noise ", pipeline, "quintic")
        assert pool == ["flip", "neighbor_noise"]

    def test_unknown_name_raises_listing_valid_names(self):
        pipeline = self._pipeline("flip")
        with pytest.raises(ValueError) as exc:
            resolve_repeat_aug_pool("nonsense", pipeline, augment_type="quintic")
        assert "nonsense" in str(exc.value)
        assert "flip" in str(exc.value)

    def test_name_not_enabled_raises_naming_the_use_flag(self):
        pipeline = self._pipeline("flip")
        with pytest.raises(ValueError) as exc:
            resolve_repeat_aug_pool("neighbor_noise", pipeline, augment_type="quintic")
        assert "--use_neighbor_noise" in str(exc.value)

    def test_empty_pipeline_raises(self):
        with pytest.raises(ValueError, match="no augmentations are enabled"):
            resolve_repeat_aug_pool("", [], augment_type="quintic")

    def test_bridge_excludes_state_perturbation_from_default_pool(self):
        """912 ms/sample per-row search: forcing it is not cost-neutral."""
        pipeline = self._pipeline("flip", "state_perturbation")
        pool, warnings = resolve_repeat_aug_pool("", pipeline, augment_type="bridge")
        assert pool == ["flip"]
        assert not any("bridge" in w for w in warnings)

    def test_bridge_default_pool_collapsing_to_flip_still_warns(self):
        """Operator did not ask for flip-alone; bridge reduced it. Warn anyway."""
        pipeline = self._pipeline("flip", "state_perturbation")
        pool, warnings = resolve_repeat_aug_pool("", pipeline, augment_type="bridge")
        assert pool == ["flip"]
        assert any("deterministic" in w for w in warnings)

    def test_bridge_allows_explicit_state_perturbation_with_warning(self):
        pipeline = self._pipeline("flip", "state_perturbation")
        pool, warnings = resolve_repeat_aug_pool(
            "state_perturbation", pipeline, augment_type="bridge"
        )
        assert pool == ["state_perturbation"]
        assert any("bridge" in w for w in warnings)

    def test_bridge_only_state_perturbation_enabled_raises(self):
        """Default pool would be empty, which is a misconfiguration, not a silent no-op."""
        pipeline = self._pipeline("state_perturbation")
        with pytest.raises(ValueError, match="empty"):
            resolve_repeat_aug_pool("", pipeline, augment_type="bridge")

    def test_quintic_keeps_state_perturbation_in_default_pool(self):
        pipeline = self._pipeline("flip", "state_perturbation")
        pool, _ = resolve_repeat_aug_pool("", pipeline, augment_type="quintic")
        assert pool == ["flip", "state_perturbation"]

    def test_flip_only_pool_warns_about_determinism(self):
        """Forced flip is deterministic: two forced occurrences mirror identically."""
        pipeline = self._pipeline("flip")
        pool, warnings = resolve_repeat_aug_pool("flip", pipeline, augment_type="quintic")
        assert pool == ["flip"]
        assert any("deterministic" in w for w in warnings)


class TestForcedAugmentationSelector:
    def _selector(self, pool=("flip", "neighbor_noise"), seed=0):
        return ForcedAugmentationSelector(list(pool), seed=seed)

    def test_empty_pool_raises(self):
        with pytest.raises(ValueError, match="pool must not be empty"):
            ForcedAugmentationSelector([])

    def test_is_bound_reflects_start_epoch(self):
        sel = self._selector()
        assert not sel.is_bound
        sel.start_epoch(0, [True], 0)
        assert sel.is_bound

    def test_exactly_one_aug_forced_per_repeat_row(self):
        sel = self._selector()
        flags = [False, True, True, False, True]
        sel.start_epoch(0, flags, 0)
        masks = sel.masks_for_batch(5, torch.device("cpu"))
        stacked = torch.stack([masks["flip"], masks["neighbor_noise"]])
        per_row = stacked.sum(dim=0)
        assert per_row.tolist() == [0, 1, 1, 0, 1]

    def test_non_repeat_rows_forced_nowhere(self):
        sel = self._selector()
        sel.start_epoch(0, [False, False, False], 0)
        masks = sel.masks_for_batch(3, torch.device("cpu"))
        for mask in masks.values():
            assert not mask.any()

    def test_mask_keys_are_exactly_the_pool(self):
        sel = self._selector(pool=("flip",))
        sel.start_epoch(0, [True, True], 0)
        masks = sel.masks_for_batch(2, torch.device("cpu"))
        assert set(masks) == {"flip"}

    def test_single_member_pool_forces_that_member(self):
        sel = self._selector(pool=("neighbor_noise",))
        sel.start_epoch(0, [True, False, True], 0)
        masks = sel.masks_for_batch(3, torch.device("cpu"))
        assert masks["neighbor_noise"].tolist() == [True, False, True]

    def test_masks_are_bool_on_requested_device(self):
        sel = self._selector()
        sel.start_epoch(0, [True], 0)
        masks = sel.masks_for_batch(1, torch.device("cpu"))
        for mask in masks.values():
            assert mask.dtype == torch.bool
            assert mask.device == torch.device("cpu")

    def test_cursor_advances_across_batches(self):
        sel = self._selector(pool=("flip",))
        sel.start_epoch(0, [True, False, False, True], 0)
        first = sel.masks_for_batch(2, torch.device("cpu"))["flip"]
        second = sel.masks_for_batch(2, torch.device("cpu"))["flip"]
        assert first.tolist() == [True, False]
        assert second.tolist() == [False, True]
        assert sel.rows_consumed == 4

    def test_cursor_handles_heterogeneous_batch_sizes(self):
        """Partial batches are the production path (drop_last=False).

        A computed j * batch_size offset passes the uniform-size tests and
        misattributes masks here, so this is what pins the running cursor.
        """
        sel = self._selector(pool=("flip",))
        sel.start_epoch(0, [True, False, False, True], 0)
        first = sel.masks_for_batch(3, torch.device("cpu"))["flip"]
        second = sel.masks_for_batch(1, torch.device("cpu"))["flip"]
        assert first.tolist() == [True, False, False]
        assert second.tolist() == [True]
        assert sel.rows_consumed == 4

    def test_reading_past_flag_list_raises(self):
        sel = self._selector()
        sel.start_epoch(0, [True, False], 0)
        with pytest.raises(RuntimeError, match="past the end"):
            sel.masks_for_batch(3, torch.device("cpu"))

    def test_start_epoch_resets_cursor(self):
        sel = self._selector(pool=("flip",))
        sel.start_epoch(0, [True, True], 0)
        sel.masks_for_batch(2, torch.device("cpu"))
        sel.start_epoch(1, [False, True], 1)
        assert sel.rows_consumed == 0
        assert sel.masks_for_batch(2, torch.device("cpu"))["flip"].tolist() == [False, True]

    def test_stale_flags_raise(self):
        """Flags stamped for a different epoch mean __iter__ has not run yet."""
        sel = self._selector()
        with pytest.raises(ValueError, match="stale"):
            sel.start_epoch(3, [True, False], 2)

    def test_missing_flags_raise(self):
        sel = self._selector()
        with pytest.raises(ValueError, match="track_repeats"):
            sel.start_epoch(0, None, None)

    def test_masks_before_start_epoch_raise(self):
        sel = self._selector()
        with pytest.raises(RuntimeError, match="start_epoch"):
            sel.masks_for_batch(2, torch.device("cpu"))

    def test_deterministic_for_same_seed_and_epoch(self):
        flags = [True] * 8
        a, b = self._selector(seed=7), self._selector(seed=7)
        a.start_epoch(2, flags, 2)
        b.start_epoch(2, flags, 2)
        assert a.masks_for_batch(8, torch.device("cpu"))["flip"].tolist() == (
            b.masks_for_batch(8, torch.device("cpu"))["flip"].tolist()
        )

    def test_choice_stream_differs_across_epochs(self):
        """A forgotten set_epoch would replay the same choices every epoch."""
        flags = [True] * 64
        sel = self._selector(seed=7)
        sel.start_epoch(0, flags, 0)
        epoch0 = sel.masks_for_batch(64, torch.device("cpu"))["flip"].tolist()
        sel.start_epoch(1, flags, 1)
        epoch1 = sel.masks_for_batch(64, torch.device("cpu"))["flip"].tolist()
        assert epoch0 != epoch1

    def test_forced_count_accumulates(self):
        sel = self._selector()
        sel.start_epoch(0, [True, False, True, True], 0)
        sel.masks_for_batch(2, torch.device("cpu"))
        sel.masks_for_batch(2, torch.device("cpu"))
        assert sel.forced_count == 3

    def test_pool_members_are_used_roughly_uniformly(self):
        """Uniform choice over the pool, not a fixed member."""
        pool = ("flip", "neighbor_noise", "state_perturbation")
        sel = self._selector(pool=pool, seed=1)
        n = 3000
        sel.start_epoch(0, [True] * n, 0)
        masks = sel.masks_for_batch(n, torch.device("cpu"))
        counts = [int(masks[name].sum()) for name in pool]
        assert sum(counts) == n
        for count in counts:
            assert 0.25 * n < count < 0.42 * n, counts


class TestValidateForcedAugFlags:
    def _args(self, **overrides):
        base = dict(
            force_aug_on_repeat=True,
            repeat_aug_pool="",
            cluster_json="/tmp/clusters.json",
            augment_type="quintic",
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def _pipeline(self, *names):
        return [(n, _FakeAug(n)) for n in AUG_ORDER if n in names]

    def test_disabled_returns_empty_pool_and_no_warnings(self):
        pool, warnings = validate_forced_aug_flags(
            self._args(force_aug_on_repeat=False), self._pipeline("flip")
        )
        assert pool == []
        assert warnings == []

    def test_requires_cluster_json(self):
        """Without the weighted sampler there are no repeat draws at all."""
        with pytest.raises(ValueError, match="--cluster_json"):
            validate_forced_aug_flags(self._args(cluster_json=""), self._pipeline("flip"))

    def test_returns_resolved_pool(self):
        pool, _ = validate_forced_aug_flags(self._args(), self._pipeline("flip", "neighbor_noise"))
        assert pool == ["flip", "neighbor_noise"]

    def test_propagates_pool_resolution_errors(self):
        with pytest.raises(ValueError, match="--use_neighbor_noise"):
            validate_forced_aug_flags(
                self._args(repeat_aug_pool="neighbor_noise"), self._pipeline("flip")
            )

    def test_propagates_warnings(self):
        _, warnings = validate_forced_aug_flags(
            self._args(repeat_aug_pool="flip"), self._pipeline("flip")
        )
        assert any("deterministic" in w for w in warnings)


class TestDuplicatePathWarning:
    def test_reports_duplicate_paths(self):
        data_list = ["/d/a.npz", "/d/b.npz", "/d/a.npz", "/d/c.npz", "/d/b.npz"]
        message = duplicate_path_warning(data_list)
        assert message is not None
        assert "2" in message

    def test_none_when_unique(self):
        assert duplicate_path_warning(["/d/a.npz", "/d/b.npz"]) is None
