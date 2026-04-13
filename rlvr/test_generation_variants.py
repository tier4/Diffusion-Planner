"""Unit tests for the generation variant registry."""

import pytest

from rlvr.generation_variants import GenerationVariant, _ALIASES, get_variant, list_variants


def test_canonical_variant_resolves():
    """Direct lookup of a registered variant returns a GenerationVariant."""
    v = get_variant("rsft_v2")
    assert isinstance(v, GenerationVariant)
    assert len(v.cl_spd_configs) > 0
    assert len(v.noise_configs) > 0
    assert v.description


def test_default_variant_resolves():
    """The fallback "default" variant must always be present."""
    v = get_variant("default")
    assert isinstance(v, GenerationVariant)
    assert len(v.cl_spd_configs) == 8


def test_aliases_resolve_to_canonical():
    """Every alias must point to a variant that exists in the registry."""
    for alias, target in _ALIASES.items():
        resolved = get_variant(alias)
        canonical = get_variant(target)
        assert resolved is canonical, f"alias {alias!r} did not resolve to {target!r}"


def test_unknown_variant_raises():
    """Unknown names must raise ValueError listing the available variants."""
    with pytest.raises(ValueError) as exc_info:
        get_variant("does_not_exist")
    msg = str(exc_info.value)
    assert "does_not_exist" in msg
    # Error message should help the caller find a real name
    assert "rsft_v2" in msg


def test_list_variants_excludes_aliases():
    """list_variants() returns only canonical names, not aliases."""
    canonical = set(list_variants())
    for alias in _ALIASES:
        assert alias not in canonical, f"alias {alias!r} leaked into list_variants()"


def test_each_variant_has_unique_labels():
    """Within a single variant, all slot labels should be unique."""
    for name in list_variants():
        v = get_variant(name)
        labels = [c["label"] for c in v.cl_spd_configs] + [c["label"] for c in v.noise_configs]
        dupes = {lbl for lbl in labels if labels.count(lbl) > 1}
        assert not dupes, f"variant {name!r} has duplicate slot labels: {sorted(dupes)}"


def test_total_slots_fit_in_K16():
    """det + cl_spd + noise must leave room for at least 0 random slots in K=16."""
    K = 16
    for name in list_variants():
        v = get_variant(name)
        used = 1 + len(v.cl_spd_configs) + len(v.noise_configs)
        assert used <= K, f"variant {name!r} uses {used} slots, exceeds K={K}"
