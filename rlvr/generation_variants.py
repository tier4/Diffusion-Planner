"""Generation variant registry for ranked SFT.

Each variant defines the 16 generation slots used per scene during ranked SFT:
  slot 0 = deterministic (always)
  slots 1..N = guided cl_spd configs (N = len(cl_spd_configs))
  slots N+1..M = noise-only configs with deterministic noise ranges
  slots M+1..15 = random pool (random CL + noise from config.noise_scale_range)

Adding a new variant:
  1. Define its `cl_spd_configs` and `noise_configs` lists.
  2. Add a `GenerationVariant(...)` entry to `_VARIANTS`.
  3. (Optional) add an alias to `_ALIASES` if backwards compat is needed.

cl_spd_config dict fields:
  cl, spd        — centerline & speed guidance scales (0 = disabled)
  noise          — (min, max) per-element noise range
  label          — human-readable identifier (used by rank_analytics)
  stretch        — speed stretch factor (default 1.0, no stretch)
  lat_eta        — lateral guidance eta in [-1, 1] (default 0, disabled)
  lat_lambda     — max lateral offset in metres (default 2.0)
  lat_scale      — lateral guidance scale (default 5.0)
  col            — collision guidance scale (default 0, disabled)

noise_config dict fields:
  noise          — (min, max) per-element noise range
  label          — human-readable identifier
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenerationVariant:
    """Defines the slot composition for a generation_variant."""
    description: str
    cl_spd_configs: list[dict]
    noise_configs: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Reusable building blocks
# ---------------------------------------------------------------------------

# Plain CL+SPD slots used in many variants
_CL5_SPD5_DET   = {"cl": 5.0,  "spd": 5.0,  "noise": (0.0, 0.0), "label": "CL5_SPD5_det"}
_CL10_SPD10_NOISY = {"cl": 10.0, "spd": 10.0, "noise": (0.5, 1.0), "label": "CL10_SPD10_noisy"}
_CL5_SPD5_NOISY = {"cl": 5.0,  "spd": 5.0,  "noise": (0.3, 0.8), "label": "CL5_SPD5_noisy"}
_CL10_SPD8_DET  = {"cl": 10.0, "spd": 8.0,  "noise": (0.0, 0.0), "label": "CL10_SPD8_det"}
_CL10_SPD8_NOISY = {"cl": 10.0, "spd": 8.0,  "noise": (0.3, 0.8), "label": "CL10_SPD8_noisy"}

# Stretched slots — winner of stretched_lateral campaign
_CL8_SPD8_STR13 = {"cl": 8.0, "spd": 8.0, "noise": (0.8, 2.5), "stretch": 1.3, "label": "CL8_SPD8_str13_n0825"}
_CL6_SPD6_STR11 = {"cl": 6.0, "spd": 6.0, "noise": (0.5, 1.5), "stretch": 1.1, "label": "CL6_SPD6_str11_n0515"}
_CL7_SPD7_STR14 = {"cl": 7.0, "spd": 7.0, "noise": (0.8, 2.0), "stretch": 1.4, "label": "CL7_SPD7_str14_n0820"}

# Lateral push slots
_LATL04 = {"cl": 5.0, "spd": 5.0, "noise": (0.3, 0.8), "lat_eta":  0.4, "lat_lambda": 2.0, "lat_scale": 5.0, "label": "CL5_SPD5_latL04"}
_LATR04 = {"cl": 5.0, "spd": 5.0, "noise": (0.3, 0.8), "lat_eta": -0.4, "lat_lambda": 2.0, "lat_scale": 5.0, "label": "CL5_SPD5_latR04"}
_LATL06 = {"cl": 5.0, "spd": 5.0, "noise": (0.5, 1.5), "lat_eta":  0.6, "lat_lambda": 2.5, "lat_scale": 5.0, "label": "CL5_SPD5_latL06_n15"}

# Common guided block used by rsft_v2 family (6 slots: 3 plain + 3 stretched)
_GUIDED_RSFT_V2 = [
    _CL5_SPD5_DET,
    _CL8_SPD8_STR13,
    _CL6_SPD6_STR11,
    _CL5_SPD5_NOISY,
    _CL7_SPD7_STR14,
    _CL10_SPD10_NOISY,
]

# Noise sweeps
_NOISE_SWEEP_FULL = [
    {"noise": (0.1, 0.3), "label": "noise_n0103"},
    {"noise": (0.3, 0.6), "label": "noise_n0306"},
    {"noise": (0.5, 1.0), "label": "noise_n0510"},
    {"noise": (0.5, 1.5), "label": "noise_n0515"},
    {"noise": (0.8, 1.8), "label": "noise_n0818"},
    {"noise": (1.0, 2.5), "label": "noise_n1025"},
    {"noise": (1.5, 3.0), "label": "noise_n1530"},
    {"noise": (2.0, 4.0), "label": "noise_n2040"},
    {"noise": (3.0, 5.0), "label": "noise_n3050"},
]

_NOISE_SWEEP_HIGH = [
    {"noise": (1.5, 3.0), "label": "noise_n1530"},
    {"noise": (2.0, 4.0), "label": "noise_n2040"},
]

_NOISE_SWEEP_MID = [
    {"noise": (0.3, 0.8), "label": "noise_n0308"},
    {"noise": (0.5, 1.5), "label": "noise_n0515"},
    {"noise": (1.0, 2.0), "label": "noise_n1020"},
    {"noise": (1.5, 3.0), "label": "noise_n1530"},
    {"noise": (2.0, 4.0), "label": "noise_n2040"},
]


# ---------------------------------------------------------------------------
# Variant registry — entries listed by category then chronological order.
# ---------------------------------------------------------------------------

_VARIANTS: dict[str, GenerationVariant] = {

    # ====== Production / current default ======
    "rsft_v2": GenerationVariant(
        description="Default RSFT base: 6 guided + 9 noise sweep (0.1->5.0).",
        cl_spd_configs=_GUIDED_RSFT_V2,
        noise_configs=_NOISE_SWEEP_FULL,
    ),

    # ====== Runner-up / honorable mentions (kept for comparison) ======
    "rsft_v2_legacy": GenerationVariant(
        description="Previous default: 6 guided + 2 fixed-noise + 7 random-CL.",
        cl_spd_configs=_GUIDED_RSFT_V2,
        noise_configs=_NOISE_SWEEP_HIGH,
    ),
    "rsft_v2_half_half": GenerationVariant(
        description="5 noise + 4 random. Aggressive — high reward/progress at the cost of lane keeping.",
        cl_spd_configs=_GUIDED_RSFT_V2,
        noise_configs=_NOISE_SWEEP_MID,
    ),
    "rsft_v2_all_random": GenerationVariant(
        description="Same guided as rsft_v2, no fixed noise, 9 random-CL slots. Conservative.",
        cl_spd_configs=_GUIDED_RSFT_V2,
        noise_configs=[],
    ),

    # ====== Original baseline (pre-rsft_v2 system) ======
    "default": GenerationVariant(
        description="Original 8 cl_spd configs (CL5/CL8/CL10 det + noisy variants), no fixed noise.",
        cl_spd_configs=[
            _CL5_SPD5_DET,
            {"cl": 8.0,  "spd": 5.0,  "noise": (0.0, 0.0), "label": "CL8_SPD5_det"},
            _CL10_SPD8_DET,
            {"cl": 10.0, "spd": 10.0, "noise": (0.0, 0.0), "label": "CL10_SPD10_det"},
            _CL5_SPD5_NOISY,
            {"cl": 8.0,  "spd": 8.0,  "noise": (0.3, 0.8), "label": "CL8_SPD8_noisy"},
            _CL10_SPD8_NOISY,
            _CL10_SPD10_NOISY,
        ],
    ),

    # ====== Historical exploratory variants (kept for reproducibility) ======
    "stretched_lateral": GenerationVariant(
        description="Pre-rsft_v2 variant: stretched + lateral push slots.",
        cl_spd_configs=[
            _CL5_SPD5_DET, _CL8_SPD8_STR13, _CL10_SPD8_DET, _LATL04,
            _CL5_SPD5_NOISY, _LATL06, _CL10_SPD8_NOISY, _CL10_SPD10_NOISY,
        ],
    ),
    "noise_swap_2": GenerationVariant(
        description="Replaces 2 CL10 slots in stretched_lateral with high-noise (n1530, n2040).",
        cl_spd_configs=[
            _CL5_SPD5_DET, _CL8_SPD8_STR13,
            {"cl": 0.0, "spd": 0.0, "noise": (1.5, 3.0), "label": "noise_n1530"},
            _LATL04, _CL5_SPD5_NOISY, _LATL06,
            {"cl": 0.0, "spd": 0.0, "noise": (2.0, 4.0), "label": "noise_n2040"},
            _CL10_SPD10_NOISY,
        ],
    ),
    "more_noise": GenerationVariant(
        description="5 noise-only slots in cl_spd positions. Drift-prone; produced rsft_v2 noise sweep idea.",
        cl_spd_configs=[
            {"cl": 0.0, "spd": 0.0, "noise": (0.3, 0.8), "label": "noise_n0308"},
            _CL8_SPD8_STR13,
            {"cl": 0.0, "spd": 0.0, "noise": (0.8, 2.0), "label": "noise_n0820"},
            {"cl": 3.0, "spd": 0.0, "noise": (0.5, 1.5), "label": "CL3_n0515"},
            _CL5_SPD5_NOISY, _LATL06,
            {"cl": 0.0, "spd": 0.0, "noise": (1.5, 3.0), "label": "noise_n1530"},
            {"cl": 0.0, "spd": 0.0, "noise": (2.0, 4.0), "label": "noise_n2040"},
        ],
    ),
    "noisy_stretched": GenerationVariant(
        description="3 stretched variants (str12/13/14) replacing CL10 redundant slots.",
        cl_spd_configs=[
            _CL5_SPD5_DET,
            {"cl": 5.0, "spd": 5.0, "noise": (0.5, 1.5), "stretch": 1.2, "label": "CL5_SPD5_str12_n0515"},
            _CL10_SPD8_DET, _CL8_SPD8_STR13, _CL5_SPD5_NOISY,
            {"cl": 5.0, "spd": 3.0, "noise": (1.0, 3.0), "stretch": 1.4, "label": "CL5_SPD3_str14_n1030"},
            _CL10_SPD8_NOISY, _CL10_SPD10_NOISY,
        ],
    ),
    "lateral": GenerationVariant(
        description="3 lateral push variants (latL04, latR04, latL06).",
        cl_spd_configs=[
            _CL5_SPD5_DET, _LATL04, _CL10_SPD8_DET, _LATR04,
            _CL5_SPD5_NOISY, _LATL06, _CL10_SPD8_NOISY, _CL10_SPD10_NOISY,
        ],
    ),
    "decoupled": GenerationVariant(
        description="CL-only and SPD-only variants (no combined CL+SPD). Each won 5-8% wins.",
        cl_spd_configs=[
            _CL5_SPD5_DET,
            {"cl": 0.0,  "spd": 5.0,  "noise": (0.3, 1.5), "label": "SPD5_only_n0315"},
            _CL10_SPD8_DET,
            {"cl": 5.0,  "spd": 0.0,  "noise": (0.3, 1.5), "label": "CL5_only_n0315"},
            _CL5_SPD5_NOISY,
            {"cl": 10.0, "spd": 0.0,  "noise": (0.5, 2.0), "label": "CL10_only_n0520"},
            _CL10_SPD8_NOISY, _CL10_SPD10_NOISY,
        ],
    ),
    "combined_winners": GenerationVariant(
        description="Top winner from each campaign (stretched + decoupled + lateral) combined.",
        cl_spd_configs=[
            _CL5_SPD5_DET, _CL8_SPD8_STR13, _CL10_SPD8_DET,
            {"cl": 5.0, "spd": 0.0, "noise": (0.3, 1.5), "label": "CL5_only_n0315"},
            _CL5_SPD5_NOISY, _LATL06, _CL10_SPD8_NOISY, _CL10_SPD10_NOISY,
        ],
    ),
    "stretched_intense": GenerationVariant(
        description="3 progressively stronger stretched variants (str12, str13, str15).",
        cl_spd_configs=[
            _CL5_SPD5_DET,
            {"cl": 5.0, "spd": 5.0, "noise": (0.5, 1.5), "stretch": 1.2, "label": "CL5_SPD5_str12_n0515"},
            _CL10_SPD8_DET, _CL8_SPD8_STR13, _CL5_SPD5_NOISY,
            {"cl": 10.0, "spd": 8.0, "noise": (1.0, 3.0), "stretch": 1.5, "label": "CL10_SPD8_str15_n1030"},
            _CL10_SPD8_NOISY, _CL10_SPD10_NOISY,
        ],
    ),
    "collision_swap": GenerationVariant(
        description="Replaces 2 CL10 slots with collision-guided variants (untested as of April 2026).",
        cl_spd_configs=[
            _CL5_SPD5_DET, _CL8_SPD8_STR13,
            {"cl": 5.0, "spd": 5.0, "noise": (0.0, 0.0), "col": 0.5, "label": "CL5_SPD5_col05_det"},
            _LATL04, _CL5_SPD5_NOISY, _LATL06,
            {"cl": 5.0, "spd": 5.0, "noise": (0.3, 0.8), "col": 1.0, "label": "CL5_SPD5_col10_n0308"},
            _CL10_SPD10_NOISY,
        ],
    ),
}


# Backwards-compatibility aliases — older configs use these names.
_ALIASES: dict[str, str] = {
    "noise_swap_2_no_lat": "rsft_v2_legacy",  # original name during exploration
    "rsft_v2_all_noise": "rsft_v2",            # equivalent to current default
}


def get_variant(name: str) -> GenerationVariant:
    """Look up a variant by name (resolves aliases)."""
    name = _ALIASES.get(name, name)
    if name not in _VARIANTS:
        available = sorted(_VARIANTS.keys())
        raise ValueError(
            f"Unknown generation_variant: {name!r}. Available: {available}"
        )
    return _VARIANTS[name]


def list_variants() -> list[str]:
    """Return canonical variant names (excludes aliases)."""
    return sorted(_VARIANTS.keys())
