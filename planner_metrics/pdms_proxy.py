"""Autoware-compatible EPDMS validation wrapper for Diffusion-Planner tensors.

The aggregation in this module mirrors
``autoware_planning_data_analyzer/src/metrics/epdms/aggregation``:

* multiplicative terms: NC * DAC * DDC * TLC
* weighted terms: EP(5) + TTC(5) + LK(2) + HC(2) + EC(2)
* fixed weighted denominator: 16
* synthetic EPDMS is available only when every required raw subscore is available
* human filtering replaces a non-EC agent subscore with 1.0 when the human
  reference also failed that subscore (abs(human) <= 1e-9)

Diffusion-Planner NPZs do not contain every Autoware ROS/lanelet input used by
``planning_data_analyzer``. Missing terms therefore remain unavailable instead of
being silently treated as neutral. A logged synthetic score is C++ comparable
only when its matching ``*_available`` flag is true; training logs use masked
means over available samples and report coverage separately.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

import planner_metrics.pdms_navsim as _ns

DT = 0.1
EPS_HUMAN = 1.0e-9

MULTIPLICATIVE_TERMS = (
    "no_at_fault_collision",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
)
WEIGHTED_TERMS = (
    ("ego_progress", 5.0),
    ("time_to_collision_within_bound", 5.0),
    ("lane_keeping", 2.0),
    ("history_comfort", 2.0),
    ("extended_comfort", 2.0),
)
SYNTHETIC_WEIGHTED_DENOMINATOR = 16.0


def _leading_shape(pred: torch.Tensor) -> tuple[int, ...]:
    return tuple(pred.shape[:-2])


def _ones_like_lead(pred: torch.Tensor) -> torch.Tensor:
    return torch.ones(_leading_shape(pred), dtype=torch.float32, device=pred.device)


def _zeros_like_lead(pred: torch.Tensor) -> torch.Tensor:
    return torch.zeros(_leading_shape(pred), dtype=torch.float32, device=pred.device)


def _tensor_from_array(values, pred: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=pred.device).reshape(
        _leading_shape(pred)
    )


def _pdms_num_threads(n_items: int) -> int:
    if n_items < 8:
        return 1
    raw = os.environ.get("PDMS_PROXY_NUM_THREADS")
    if raw is not None:
        try:
            return max(1, min(int(raw), n_items))
        except ValueError:
            return 1
    return max(1, min(8, n_items))


def _parallel_list(fn, n_items: int) -> list:
    workers = _pdms_num_threads(n_items)
    if workers <= 1:
        return [fn(i) for i in range(n_items)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, range(n_items)))


def _available(pred: torch.Tensor, value: bool) -> torch.Tensor:
    fill = 1.0 if value else 0.0
    return torch.full(_leading_shape(pred), fill, dtype=torch.float32, device=pred.device)


def _set_metric(
    out: dict[str, torch.Tensor], name: str, value: torch.Tensor, available=True
) -> None:
    out[name] = value.float()
    if torch.is_tensor(available):
        out[f"{name}_available"] = available.to(device=value.device, dtype=torch.float32)
    else:
        out[f"{name}_available"] = torch.full_like(value.float(), 1.0 if available else 0.0)


def _metric_or_zero(metrics: dict[str, torch.Tensor], key: str, ref: torch.Tensor) -> torch.Tensor:
    value = metrics.get(key)
    if value is None:
        return torch.zeros_like(ref)
    return value.to(device=ref.device, dtype=torch.float32)


def _availability(metrics: dict[str, torch.Tensor], key: str, ref: torch.Tensor) -> torch.Tensor:
    value = metrics.get(f"{key}_available")
    if value is None:
        return torch.zeros_like(ref)
    return value.to(device=ref.device, dtype=torch.float32)


def comfort_score(traj: torch.Tensor, dt: float = DT) -> torch.Tensor:
    """NAVSIM/C++ comfort-threshold score in {0, 1} for a trajectory tensor."""
    arr = traj.detach().to(torch.float64).cpu().numpy()
    out = _ns.comfort_score(arr, dt)
    return torch.as_tensor(np.asarray(out), dtype=torch.float32, device=traj.device)


def _ego_progress_and_gate(
    pred: torch.Tensor, gt: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    lead = pred.shape[:-2]
    T = pred.shape[-2]
    pred_n = pred.detach().to(torch.float64).cpu().numpy().reshape(-1, T, 4)
    gt_n = gt.detach().to(torch.float64).cpu().numpy().reshape(-1, T, 4)
    pairs = [_ns.ego_progress_with_gate(pred_n[i], gt_n[i]) for i in range(pred_n.shape[0])]
    ep = np.asarray([p[0] for p in pairs], dtype=np.float64)
    gated = np.asarray([float(p[1]) for p in pairs], dtype=np.float64)
    dev = pred.device
    return (
        torch.as_tensor(ep.reshape(lead), dtype=torch.float32, device=dev),
        torch.as_tensor(gated.reshape(lead), dtype=torch.float32, device=dev),
    )


def ego_progress_score(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return _ego_progress_and_gate(pred, gt)[0]


@dataclass(frozen=True)
class SyntheticEpdms:
    raw_available: torch.Tensor
    raw_multiplicative_metrics_prod: torch.Tensor
    raw_weighted_metrics: torch.Tensor
    raw: torch.Tensor
    human_filtered_available: torch.Tensor
    human_filtered_multiplicative_metrics_prod: torch.Tensor
    human_filtered_weighted_metrics: torch.Tensor
    human_filtered: torch.Tensor


@dataclass(frozen=True)
class ProxyEpdms:
    available: torch.Tensor
    availability_ratio: torch.Tensor
    multiplicative_metrics_prod: torch.Tensor
    weighted_metrics: torch.Tensor
    total: torch.Tensor


def proxy_epdms(agent: dict[str, torch.Tensor]) -> ProxyEpdms:
    """Best-effort EPDMS proxy over metrics that are actually available.

    This is intentionally *not* the strict Autoware/C++ synthetic EPDMS. The
    strict score requires every EPDMS input, including traffic-light compliance
    and extended comfort, and remains exposed as ``synthetic_epdms_raw`` with an
    availability flag.

    For training-time model selection we still need a scalar planning-quality
    proxy on current Diffusion-Planner NPZs. This proxy therefore:

    * multiplies available multiplicative terms and treats missing terms as
      "not evaluated" rather than as 0;
    * averages available weighted terms using the sum of available weights as
      the denominator;
    * reports availability coverage so dashboards can show how much of the full
      EPDMS formula was actually evaluated.
    """
    ref = agent["ego_progress"].float()

    mult = torch.ones_like(ref)
    available_count = torch.zeros_like(ref)
    for key in MULTIPLICATIVE_TERMS:
        av = _availability(agent, key, ref)
        value = _metric_or_zero(agent, key, ref)
        mult = mult * torch.where(av > 0.5, value, torch.ones_like(value))
        available_count = available_count + av

    weighted_num = torch.zeros_like(ref)
    weighted_den = torch.zeros_like(ref)
    total_term_count = float(len(MULTIPLICATIVE_TERMS) + len(WEIGHTED_TERMS))
    for key, weight in WEIGHTED_TERMS:
        av = _availability(agent, key, ref)
        value = _metric_or_zero(agent, key, ref)
        weighted_num = weighted_num + weight * torch.where(av > 0.5, value, torch.zeros_like(value))
        weighted_den = weighted_den + weight * av
        available_count = available_count + av

    available = weighted_den > 0.0
    weighted = torch.where(
        available, weighted_num / torch.clamp_min(weighted_den, 1.0), torch.zeros_like(ref)
    )
    total = mult * weighted
    availability_ratio = available_count / total_term_count
    return ProxyEpdms(
        available=available.float(),
        availability_ratio=availability_ratio,
        multiplicative_metrics_prod=mult,
        weighted_metrics=weighted,
        total=total,
    )


def synthetic_epdms(
    agent: dict[str, torch.Tensor], human: dict[str, torch.Tensor] | None = None
) -> SyntheticEpdms:
    """C++-aligned synthetic EPDMS aggregation.

    ``agent`` and optional ``human`` are dictionaries with raw subscore tensors and
    ``<metric>_available`` tensors. Missing availability means unavailable.
    """
    ref = agent["ego_progress"].float()

    raw_mult_available = torch.ones_like(ref)
    raw_mult = torch.ones_like(ref)
    for key in MULTIPLICATIVE_TERMS:
        raw_mult_available = raw_mult_available * _availability(agent, key, ref)
        raw_mult = raw_mult * _metric_or_zero(agent, key, ref)

    raw_weighted_available = torch.ones_like(ref)
    raw_weighted_num = torch.zeros_like(ref)
    for key, weight in WEIGHTED_TERMS:
        raw_weighted_available = raw_weighted_available * _availability(agent, key, ref)
        raw_weighted_num = raw_weighted_num + weight * _metric_or_zero(agent, key, ref)

    raw_available = raw_mult_available * raw_weighted_available
    raw_weighted = raw_weighted_num / SYNTHETIC_WEIGHTED_DENOMINATOR
    raw = raw_mult * raw_weighted

    if human is None:
        return SyntheticEpdms(
            raw_available=raw_available,
            raw_multiplicative_metrics_prod=raw_mult,
            raw_weighted_metrics=raw_weighted,
            raw=raw,
            human_filtered_available=torch.zeros_like(ref),
            human_filtered_multiplicative_metrics_prod=torch.zeros_like(ref),
            human_filtered_weighted_metrics=torch.zeros_like(ref),
            human_filtered=torch.zeros_like(ref),
        )

    hf_mult_available = torch.ones_like(ref)
    hf_mult = torch.ones_like(ref)
    for key in MULTIPLICATIVE_TERMS:
        a = _metric_or_zero(agent, key, ref)
        h = _metric_or_zero(human, key, ref)
        a_av = _availability(agent, key, ref)
        h_av = _availability(human, key, ref)
        applied = (a_av > 0.5) & (h_av > 0.5) & (h.abs() <= EPS_HUMAN)
        hf_mult_available = hf_mult_available * a_av
        hf_mult = hf_mult * torch.where(applied, torch.ones_like(a), a)

    hf_weighted_available = torch.ones_like(ref)
    hf_weighted_num = torch.zeros_like(ref)
    for key, weight in WEIGHTED_TERMS:
        a = _metric_or_zero(agent, key, ref)
        a_av = _availability(agent, key, ref)
        hf_weighted_available = hf_weighted_available * a_av
        if key == "extended_comfort":
            filtered = a
        else:
            h = _metric_or_zero(human, key, ref)
            h_av = _availability(human, key, ref)
            applied = (a_av > 0.5) & (h_av > 0.5) & (h.abs() <= EPS_HUMAN)
            filtered = torch.where(applied, torch.ones_like(a), a)
        hf_weighted_num = hf_weighted_num + weight * filtered

    hf_available = hf_mult_available * hf_weighted_available
    hf_weighted = hf_weighted_num / SYNTHETIC_WEIGHTED_DENOMINATOR
    hf = hf_mult * hf_weighted

    return SyntheticEpdms(
        raw_available=raw_available,
        raw_multiplicative_metrics_prod=raw_mult,
        raw_weighted_metrics=raw_weighted,
        raw=raw,
        human_filtered_available=hf_available,
        human_filtered_multiplicative_metrics_prod=hf_mult,
        human_filtered_weighted_metrics=hf_weighted,
        human_filtered=hf,
    )


def add_synthetic_epdms(
    out: dict[str, torch.Tensor], human: dict[str, torch.Tensor] | None = None
) -> dict[str, torch.Tensor]:
    synthetic = synthetic_epdms(out, human)
    proxy = proxy_epdms(out)
    out["synthetic_epdms_raw_available"] = synthetic.raw_available
    out["synthetic_epdms_raw_multiplicative_metrics_prod"] = (
        synthetic.raw_multiplicative_metrics_prod
    )
    out["synthetic_epdms_raw_weighted_metrics"] = synthetic.raw_weighted_metrics
    out["synthetic_epdms_raw"] = synthetic.raw
    out["synthetic_epdms_human_filtered_available"] = synthetic.human_filtered_available
    out["synthetic_epdms_human_filtered_multiplicative_metrics_prod"] = (
        synthetic.human_filtered_multiplicative_metrics_prod
    )
    out["synthetic_epdms_human_filtered_weighted_metrics"] = (
        synthetic.human_filtered_weighted_metrics
    )
    out["synthetic_epdms_human_filtered"] = synthetic.human_filtered
    out["proxy_epdms_available"] = proxy.available
    out["proxy_epdms_availability_ratio"] = proxy.availability_ratio
    out["proxy_epdms_multiplicative_metrics_prod"] = proxy.multiplicative_metrics_prod
    out["proxy_epdms_weighted_metrics"] = proxy.weighted_metrics
    out["proxy_epdms"] = proxy.total
    # Dashboard-facing final score: usable best-effort proxy for current DP NPZs.
    # Strict Autoware/C++ comparability remains exposed by synthetic_epdms_raw(_available).
    out["total"] = proxy.total
    out["total_available"] = proxy.available
    return out


def pdms_proxy(
    pred: torch.Tensor,
    gt: torch.Tensor,
    agent_boxes_per_t: list | None = None,
    ego_dims: tuple[float, float] | np.ndarray | None = None,
    agent_labels_per_t: list | None = None,
    static_labels: set | None = None,
    border_lines: list | None = None,
    route_polys: list | None = None,
    lane_rings: list | None = None,
    intersection_rings: list | None = None,
    route_centerlines: list | None = None,
    lk_exempt: list | None = None,
    drivable_area_compliance_values: np.ndarray | None = None,
    extended_comfort_values: np.ndarray | None = None,
    traffic_light_compliance_values: np.ndarray | None = None,
    precomputed_states: np.ndarray | None = None,
    add_aggregation: bool = True,
    self_reference_progress: bool = False,
    dt: float = DT,
) -> dict[str, torch.Tensor]:
    """Compute raw EPDMS subscores available from DP tensors.

    The returned keys use Autoware planning_data_analyzer names plus
    ``<metric>_available`` flags. Missing DP inputs leave the corresponding metric
    unavailable. ``add_synthetic_epdms`` performs the strict C++ aggregation.
    """
    out: dict[str, torch.Tensor] = {}

    pred_n_cache = None
    states_n_cache = None

    def _pred_numpy() -> np.ndarray:
        nonlocal pred_n_cache
        if pred_n_cache is None:
            T = pred.shape[-2]
            pred_n_cache = pred.detach().to(torch.float64).cpu().numpy().reshape(-1, T, 4)
        return pred_n_cache

    def _states_numpy() -> np.ndarray:
        nonlocal states_n_cache
        if states_n_cache is None:
            T = pred.shape[-2]
            if precomputed_states is None:
                states_n_cache = _ns.states_from_poses(_pred_numpy(), dt)
            else:
                states_n_cache = np.asarray(precomputed_states, dtype=np.float64).reshape(
                    -1, T, _ns.STATE_SIZE
                )
        return states_n_cache

    if self_reference_progress:
        ep = _ones_like_lead(pred)
        ep_gated = _zeros_like_lead(pred)
    else:
        ep, ep_gated = _ego_progress_and_gate(pred, gt)
    _set_metric(out, "ego_progress", ep, True)
    out["ego_progress_gt_gate"] = ep_gated

    hc = torch.as_tensor(
        _ns.comfort_score_from_states(_states_numpy(), dt).reshape(_leading_shape(pred)),
        dtype=torch.float32,
        device=pred.device,
    )
    _set_metric(out, "history_comfort", hc, True)

    dac_arr = None
    if drivable_area_compliance_values is not None:
        dac_arr = np.asarray(drivable_area_compliance_values, dtype=np.float64).reshape(-1).copy()
    if (
        border_lines is not None
        and ego_dims is not None
        and (dac_arr is None or np.isnan(dac_arr).any())
    ):
        T = pred.shape[-2]
        pred_n = _pred_numpy()
        dims = np.asarray(ego_dims, dtype=np.float64)
        if dims.ndim == 1:
            dims = np.broadcast_to(dims, (pred_n.shape[0], dims.shape[0]))
        def _dac_one(i: int) -> float:
            if dac_arr is not None and not np.isnan(dac_arr[i]):
                return float(dac_arr[i])
            if dims.shape[1] == 3:
                offset, length, width = (
                    float(dims[i][0]) / 2.0,
                    float(dims[i][1]),
                    float(dims[i][2]),
                )
            else:
                offset, length, width = 0.0, float(dims[i][0]), float(dims[i][1])
            return _ns.dac_from_road_borders(
                pred_n[i], border_lines[i], length, width, center_offset=offset
            )

        values = _parallel_list(_dac_one, pred_n.shape[0])
        dac_arr = np.asarray(values, dtype=np.float64)
    if dac_arr is not None and not np.isnan(dac_arr).any():
        _set_metric(out, "drivable_area_compliance", _tensor_from_array(dac_arr, pred), True)

    if route_polys is not None:
        T = pred.shape[-2]
        pred_n = _pred_numpy()
        ddc = np.asarray(
            _parallel_list(
                lambda i: _ns.ddc_from_route_lanes(pred_n[i], route_polys[i], dt),
                pred_n.shape[0],
            ),
            dtype=np.float64,
        )
        _set_metric(out, "driving_direction_compliance", _tensor_from_array(ddc, pred), True)

    if route_centerlines is not None:
        T = pred.shape[-2]
        pred_n = _pred_numpy()
        lk = np.asarray(
            _parallel_list(
                lambda i: _ns.lane_keeping_score(
                    pred_n[i],
                    route_centerlines[i],
                    [] if intersection_rings is None else intersection_rings[i],
                    dt,
                    lane_change_exempt=bool(lk_exempt[i]) if lk_exempt is not None else False,
                ),
                pred_n.shape[0],
            ),
            dtype=np.float64,
        )
        _set_metric(out, "lane_keeping", _tensor_from_array(lk, pred), True)

    if extended_comfort_values is not None:
        ec = np.asarray(extended_comfort_values, dtype=np.float64).reshape(-1)
        _set_metric(
            out,
            "extended_comfort",
            _tensor_from_array(np.nan_to_num(ec, nan=0.0), pred),
            ~torch.isnan(_tensor_from_array(ec, pred)),
        )

    if traffic_light_compliance_values is not None:
        tlc = np.asarray(traffic_light_compliance_values, dtype=np.float64).reshape(-1)
        _set_metric(
            out,
            "traffic_light_compliance",
            _tensor_from_array(np.nan_to_num(tlc, nan=0.0), pred),
            ~torch.isnan(_tensor_from_array(tlc, pred)),
        )

    if agent_boxes_per_t is not None and ego_dims is not None:
        lead = pred.shape[:-2]
        T = pred.shape[-2]
        pred_n = _pred_numpy()
        states_n = _states_numpy()
        dims = np.asarray(ego_dims, dtype=np.float64)
        if dims.ndim == 1:
            dims = np.broadcast_to(dims, (pred_n.shape[0], dims.shape[0]))
        def _collision_one(i: int) -> tuple[float, float]:
            if dims.shape[1] == 3:
                offset, length, width = (
                    float(dims[i][0]) / 2.0,
                    float(dims[i][1]),
                    float(dims[i][2]),
                )
            else:
                offset, length, width = 0.0, float(dims[i][0]), float(dims[i][1])
            states = states_n[i]
            area_flags = None
            if lane_rings is not None or intersection_rings is not None:
                area_flags = _ns.ego_area_flags(
                    states,
                    [] if lane_rings is None else lane_rings[i],
                    [] if intersection_rings is None else intersection_rings[i],
                    length,
                    width,
                    center_offset=offset,
                )
            boxes_t = list(agent_boxes_per_t[i])[:T]
            labels_t = list(agent_labels_per_t[i])[:T] if agent_labels_per_t is not None else None
            nc = _ns.no_at_fault_collision(
                states,
                boxes_t,
                length,
                width,
                agent_labels_per_t=labels_t,
                static_labels=static_labels,
                center_offset=offset,
                area_flags=area_flags,
            )
            ttc = _ns.time_to_collision(
                states, boxes_t, length, width, dt, center_offset=offset, area_flags=area_flags
            )
            return nc, ttc

        nc_ttc = _parallel_list(_collision_one, pred_n.shape[0])
        nc_list = [x[0] for x in nc_ttc]
        ttc_list = [x[1] for x in nc_ttc]
        dev = pred.device
        _set_metric(
            out,
            "no_at_fault_collision",
            torch.as_tensor(np.asarray(nc_list).reshape(lead), dtype=torch.float32, device=dev),
            True,
        )
        _set_metric(
            out,
            "time_to_collision_within_bound",
            torch.as_tensor(np.asarray(ttc_list).reshape(lead), dtype=torch.float32, device=dev),
            True,
        )

    # Backward-compatible aliases for older dashboards. These are raw subscore
    # aliases, not synthetic EPDMS.
    alias_pairs = {
        "comfort": "history_comfort",
        "ttc": "time_to_collision_within_bound",
        "dac": "drivable_area_compliance",
        "ddc": "driving_direction_compliance",
        "no_collision": "no_at_fault_collision",
    }
    for alias, key in alias_pairs.items():
        if key in out:
            out[alias] = out[key]
            out[f"{alias}_available"] = out[f"{key}_available"]

    if add_aggregation:
        add_synthetic_epdms(out)
    return out


def pdms_proxy_masked(
    pred: torch.Tensor,
    gt: torch.Tensor,
    agent_boxes_per_t: list,
    agent_labels_per_t: list | None = None,
    ego_dims: np.ndarray | None = None,
    available: list | None = None,
    static_labels: set | None = None,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """Masked wrapper for batches with partial object-box availability."""
    B = pred.shape[0]
    avail = [True] * B if available is None else [bool(a) for a in available]
    ai = [i for i, a in enumerate(avail) if a]
    ui = [i for i, a in enumerate(avail) if not a]
    if not ui:
        return pdms_proxy(
            pred,
            gt,
            agent_boxes_per_t=agent_boxes_per_t,
            ego_dims=ego_dims,
            agent_labels_per_t=agent_labels_per_t,
            static_labels=static_labels,
            **kwargs,
        )
    p_all = pdms_proxy(pred, gt, ego_dims=ego_dims, **kwargs)
    if ai:
        p_av = pdms_proxy(
            pred[ai],
            gt[ai],
            agent_boxes_per_t=[agent_boxes_per_t[i] for i in ai],
            ego_dims=None if ego_dims is None else np.asarray(ego_dims)[ai],
            agent_labels_per_t=None
            if agent_labels_per_t is None
            else [agent_labels_per_t[i] for i in ai],
            static_labels=static_labels,
            **{
                k: ([v[i] for i in ai] if isinstance(v, list) and len(v) == B else v)
                for k, v in kwargs.items()
            },
        )
        for key, val in p_av.items():
            if key not in p_all or p_all[key].shape[0] != B:
                continue
            p_all[key][ai] = val
    return p_all


def epdms_human_filtered(
    agent: dict[str, torch.Tensor], human: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Return the C++ human-filtered synthetic EPDMS tensor."""
    return synthetic_epdms(agent, human).human_filtered


def pdms_proxy_modes(
    pred_modes: torch.Tensor, gt: torch.Tensor, chosen_idx: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """Score each mode with the strict raw synthetic EPDMS when available."""
    b, m = pred_modes.shape[0], pred_modes.shape[1]
    per_mode = torch.stack(
        [pdms_proxy(pred_modes[:, i], gt)["synthetic_epdms_raw"] for i in range(m)], dim=1
    )
    oracle = per_mode.max(dim=1).values
    if chosen_idx is None:
        chosen = per_mode[:, 0]
    else:
        chosen = per_mode[torch.arange(b, device=per_mode.device), chosen_idx]
    return {"chosen": chosen, "oracle": oracle, "per_mode": per_mode}
