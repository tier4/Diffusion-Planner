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

# Very strong CL + stretch slots for curve-recovery exploration.
_CL20_SPD10_STR13 = {"cl": 20.0, "spd": 10.0, "noise": (0.8, 2.0), "stretch": 1.3, "label": "CL20_SPD10_str13_n0820"}
_CL30_SPD15_STR15 = {"cl": 30.0, "spd": 15.0, "noise": (1.0, 2.5), "stretch": 1.5, "label": "CL30_SPD15_str15_n1025"}

_CL12_SPD8_STR12 = {"cl": 12.0, "spd": 8.0, "noise": (0.5, 1.5), "stretch": 1.2, "label": "CL12_SPD8_str12_n0515"}
_CL15_SPD10_STR13 = {"cl": 15.0, "spd": 10.0, "noise": (0.8, 2.0), "stretch": 1.3, "label": "CL15_SPD10_str13_n0820"}

# Low-noise clean variants — SFT targets that transfer cleanly to deterministic val.
# Same CL strengths but noise_max <= 1.0 so the winner trajectory isn't noise-contaminated.
_CL10_SPD8_DET = {"cl": 10.0, "spd": 8.0, "noise": (0.0, 0.0), "label": "CL10_SPD8_det"}
_CL12_SPD8_STR12_CLEAN = {"cl": 12.0, "spd": 8.0, "noise": (0.3, 0.8), "stretch": 1.2, "label": "CL12_SPD8_str12_n0308"}
_CL15_SPD10_STR13_CLEAN = {"cl": 15.0, "spd": 10.0, "noise": (0.3, 0.8), "stretch": 1.3, "label": "CL15_SPD10_str13_n0308"}

# Clean low-noise slots for route-guidance variant (cl ≤ 10, noise max ≤ 1.0).
_CL7_SPD7_LOWNOISE = {"cl": 7.0, "spd": 7.0, "noise": (0.3, 0.8), "label": "CL7_SPD7_n0308"}
_CL8_SPD8_LOWNOISE = {"cl": 8.0, "spd": 8.0, "noise": (0.5, 1.0), "label": "CL8_SPD8_n0510"}

# Low-scale route-lanes-centerline (rl_cl) sweep slots. scale ≤ 3.0 with
# matching spd, no stretch. Paired with use_route_cl_guidance=True to target
# route_lanes centerline instead of the nearest-lane centerline.
_RL_CL_1_5_SPD1_5_DET   = {"cl": 1.5, "spd": 1.5, "noise": (0.0, 0.0), "label": "RL_CL1.5_SPD1.5_det"}
_RL_CL_2_0_SPD2_0_DET   = {"cl": 2.0, "spd": 2.0, "noise": (0.0, 0.0), "label": "RL_CL2.0_SPD2.0_det"}
_RL_CL_2_5_SPD2_5_DET   = {"cl": 2.5, "spd": 2.5, "noise": (0.0, 0.0), "label": "RL_CL2.5_SPD2.5_det"}
_RL_CL_3_0_SPD3_0_DET   = {"cl": 3.0, "spd": 3.0, "noise": (0.0, 0.0), "label": "RL_CL3.0_SPD3.0_det"}
_RL_CL_2_0_SPD2_0_NOISY = {"cl": 2.0, "spd": 2.0, "noise": (0.3, 0.8), "label": "RL_CL2.0_SPD2.0_n0308"}
_RL_CL_2_5_SPD2_5_NOISY = {"cl": 2.5, "spd": 2.5, "noise": (0.3, 0.8), "label": "RL_CL2.5_SPD2.5_n0308"}
_RL_CL_3_0_SPD3_0_NOISY = {"cl": 3.0, "spd": 3.0, "noise": (0.3, 0.8), "label": "RL_CL3.0_SPD3.0_n0308"}

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
    "strong_cl_stretch": GenerationVariant(
        description="rsft_v2 guided + 2 very-strong CL+stretch slots (cl=20/30). "
                    "Replaces 2 low-noise slots to keep K=16.",
        cl_spd_configs=_GUIDED_RSFT_V2 + [_CL20_SPD10_STR13, _CL30_SPD15_STR15],
        noise_configs=_NOISE_SWEEP_FULL[2:],  # drop n0103 and n0306 to keep K=16 (6+2 guided + 7 noise + det = 16)
    ),
    "strong_cl_stretch_v2": GenerationVariant(
        description="rsft_v2 guided + 2 mid-strong CL+stretch slots (cl=12 / cl=15).",
        cl_spd_configs=_GUIDED_RSFT_V2 + [_CL12_SPD8_STR12, _CL15_SPD10_STR13],
        noise_configs=_NOISE_SWEEP_FULL[2:],
    ),
    "route_cl_low_noise": GenerationVariant(
        description="Route-centerline guidance (uses route_lanes matching reward) + "
                    "ALL slots have noise_max <= 1.0 + cl values capped at 10. "
                    "Requires use_route_cl_guidance=True in config to swap "
                    "centerline_following → route_centerline_following. K=8: "
                    "1 det + 6 route-guided + 1 noise-only-low-noise.",
        cl_spd_configs=[
            _CL5_SPD5_DET,           # cl=5, no noise
            _CL5_SPD5_NOISY,         # cl=5, noise (0.3, 0.8)
            _CL7_SPD7_LOWNOISE,      # cl=7, noise (0.3, 0.8)
            _CL8_SPD8_LOWNOISE,      # cl=8, noise (0.5, 1.0)
            _CL10_SPD8_DET,          # cl=10, no noise
            _CL10_SPD10_NOISY,       # cl=10, noise (0.5, 1.0)
        ],
        noise_configs=[
            {"noise": (0.5, 1.0), "label": "noise_n0510"},  # only low-noise exploration slot
        ],
    ),
    "rl_cl_soft_sweep": GenerationVariant(
        description="Low-scale CL sweep (scale ∈ {1.5, 2.0, 2.5, 3.0}) with no "
                    "pure-noise slots. K=8: 1 det + 4 det-sweep + 3 low-noise-sweep. "
                    "Targets route_lanes centerline when the caller passes "
                    "use_route_cl_guidance=True (otherwise these slots emit the "
                    "legacy nearest-lane centerline_following guidance).",
        cl_spd_configs=[
            _RL_CL_1_5_SPD1_5_DET,       # rl_cl=1.5, no noise
            _RL_CL_2_0_SPD2_0_DET,       # rl_cl=2.0, no noise
            _RL_CL_2_5_SPD2_5_DET,       # rl_cl=2.5, no noise
            _RL_CL_3_0_SPD3_0_DET,       # rl_cl=3.0, no noise
            _RL_CL_2_0_SPD2_0_NOISY,     # rl_cl=2.0, noise (0.3, 0.8)
            _RL_CL_2_5_SPD2_5_NOISY,     # rl_cl=2.5, noise (0.3, 0.8)
            _RL_CL_3_0_SPD3_0_NOISY,     # rl_cl=3.0, noise (0.3, 0.8)
        ],
        noise_configs=[],
    ),
    "clean_cl_only": GenerationVariant(
        description="Only clean CL-guided slots — no pure-noise, no high-noise variants. "
                    "All cl_spd slots have noise_max <= 1.0 so SFT-to-winner produces "
                    "trajectories that transfer cleanly to deterministic val. "
                    "K = 1 det + 7 clean guided = 8. CL range 5-15.",
        cl_spd_configs=[
            _CL5_SPD5_DET,              # cl=5, noise=(0,0)          — pure CL det
            _CL5_SPD5_NOISY,            # cl=5, noise=(0.3,0.8)      — low noise
            _CL10_SPD8_DET,             # cl=10, noise=(0,0)          — mid CL det
            _CL10_SPD10_NOISY,          # cl=10, noise=(0.5,1.0)      — mid CL low-noise
            _CL6_SPD6_STR11,            # cl=6, stretch=1.1, noise=(0.5,1.5)  <-- kept at 1.5 max (borderline)
            _CL12_SPD8_STR12_CLEAN,     # cl=12, stretch=1.2, noise=(0.3,0.8)
            _CL15_SPD10_STR13_CLEAN,    # cl=15, stretch=1.3, noise=(0.3,0.8)
        ],
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
    "rsft_v2_col4": GenerationVariant(
        description=(
            "Four collision-guided slots for the static-collision audit: "
            "collision scale 0.5 (det noise), 1.0, 2.0, 1.5 with varied "
            "noise. Rest of the 16 slots: 1 pure det (slot 0) + 2 non-"
            "collision CL+SPD guided (str13, noisy) + 9 noise sweep. "
            "Total 16. Used to mine / probe J6 scenes where the base "
            "model drives through parked cars and see whether any of the "
            "4 collision slots rescue the scene."
        ),
        cl_spd_configs=[
            {"cl": 5.0, "spd": 5.0, "noise": (0.0, 0.0), "col": 0.5, "label": "CL5_SPD5_col05_det"},
            {"cl": 5.0, "spd": 5.0, "noise": (0.3, 0.8), "col": 1.0, "label": "CL5_SPD5_col10_n0308"},
            {"cl": 5.0, "spd": 5.0, "noise": (0.5, 1.5), "col": 2.0, "label": "CL5_SPD5_col20_n0515"},
            {"cl": 10.0, "spd": 8.0, "noise": (0.5, 1.5), "col": 1.5, "label": "CL10_SPD8_col15_n0515"},
            _CL8_SPD8_STR13,
            _CL10_SPD10_NOISY,
        ],
        noise_configs=_NOISE_SWEEP_FULL,
    ),
}


# Backwards-compatibility aliases — older configs use these names.
_ALIASES: dict[str, str] = {
    "noise_swap_2_no_lat": "rsft_v2_legacy",  # original name during exploration
    "rsft_v2_all_noise": "rsft_v2",            # equivalent to current default
    "route_cl_soft_sweep": "rl_cl_soft_sweep", # renamed 2026-04-24 → rl_cl naming
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


def clean_slot_mask(variant: str, K: int, noise_max_threshold: float = 1.0) -> list[bool]:
    """Return a K-length bool mask of slots suitable as clean SFT targets.

    A slot is "clean" when it carries useful CL signal without heavy noise
    contamination — so that MSE SFT against the winner trajectory actually
    transfers to a deterministic val-time output. Criteria:
      * slot 0 (det_pure): always clean
      * cl_spd slots with cl>0 AND noise_max <= threshold: clean
      * pure-noise slots: NOT clean (noise dominates the signal)
      * random filler slots: NOT clean

    Used by grpo_sft_trainer when `sft_target_clean_slots_only=True` in config.
    """
    v = get_variant(variant)
    n_cl = len(v.cl_spd_configs)
    n_noise = len(v.noise_configs)
    mask = [False] * K
    mask[0] = True  # det_pure is always clean
    for i, c in enumerate(v.cl_spd_configs, start=1):
        if i >= K:
            break
        noise = c.get("noise", (0.0, 0.0))
        noise_max = float(noise[1]) if len(noise) >= 2 else 0.0
        cl_val = float(c.get("cl", 0.0))
        if cl_val > 0 and noise_max <= noise_max_threshold:
            mask[i] = True
    # cl_spd_configs slice ends at 1+n_cl; noise slots and randoms are NOT clean
    return mask
