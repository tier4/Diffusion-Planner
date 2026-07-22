"""PlanTF decoder head: forward contract, winner-takes-all loss, and wiring."""

from argparse import Namespace

import pytest
import torch
from diffusion_planner.dimensions import (
    MAX_NUM_NEIGHBORS,
    OUTPUT_T,
    POSE_DIM,
    TURN_INDICATOR_OUTPUT_DIM,
)
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.model.module.plantf_decoder import (
    PlanTFTrajectoryHead,
    compute_plantf_training_loss,
)
from diffusion_planner.utils.normalizer import StateNormalizer
from diffusion_planner.utils.onnx_export import (
    FULL_INPUT_NAMES,
    FULL_OUTPUT_NAMES,
    FullONNXWrapper,
    PlanTFFullONNXWrapper,
    build_dummy_inputs,
    build_wrappers,
    export_onnx,
    onnx_export_backends,
)
from diffusion_planner.validate_model import compute_plantf_mode_metrics

NUM_MODES = 3
HIDDEN_DIM = 32


def _state_normalizer(num_agents):
    mean = torch.zeros(num_agents, 1, POSE_DIM)
    mean[:, :, 0] = 1.0  # non-trivial mean so denormalization is exercised
    std = torch.full((num_agents, 1, POSE_DIM), 2.0)
    return StateNormalizer(mean, std)


def _config():
    return Namespace(
        # encoder
        hidden_dim=HIDDEN_DIM,
        use_ego_history=True,
        ego_history_dropout_rate=0.0,
        use_turn_indicators=True,
        agent_num=MAX_NUM_NEIGHBORS,
        static_objects_num=5,
        static_objects_state_dim=10,
        lane_num=140,
        lane_len=20,
        route_num=25,
        route_len=20,
        polygon_num=10,
        polygon_len=40,
        line_string_num=60,
        line_string_len=20,
        time_len=31,
        encoder_drop_path_rate=0.0,
        encoder_mixer_depth=2,
        encoder_fusion_depth=2,
        num_heads=2,
        # decoder
        decoder_type="plantf",
        num_modes=NUM_MODES,
        predicted_neighbor_num=MAX_NUM_NEIGHBORS,
        future_len=OUTPUT_T,
        use_velocity_representation=False,
        state_normalizer=_state_normalizer(1 + MAX_NUM_NEIGHBORS),
    )


def _inputs(batch_size=2):
    torch.manual_seed(0)
    return {
        k: v.repeat(batch_size, *([1] * (v.dim() - 1))) for k, v in build_dummy_inputs().items()
    }


def _loss_args(config):
    return Namespace(
        decoder_type="plantf",
        state_normalizer=config.state_normalizer,
        observation_normalizer=None,
        ego_prediction_horizon=OUTPUT_T,
        coeff_velocity=1.0,
        coeff_timestep=[1.0, 1.0, 1.0, 1.0],
        coeff_position_lat_loss=1.0,
        coeff_position_lon_loss=1.0,
        coeff_heading_l2_loss=1.0,
        coeff_road_border_loss=0.0,
        coeff_neighbor_collision_loss=0.0,
        road_border_margin=0.25,
        road_border_n_interp=2,
        neighbor_collision_margin_vehicle=0.25,
        neighbor_collision_margin_pedestrian=1.0,
        neighbor_collision_margin_bicycle=0.5,
    )


def test_trajectory_head_shapes():
    torch.manual_seed(0)
    head = PlanTFTrajectoryHead(
        embed_dim=HIDDEN_DIM, num_modes=NUM_MODES, future_steps=OUTPUT_T, out_channels=4
    )
    loc, pi = head(torch.randn(2, HIDDEN_DIM))
    assert loc.shape == (2, NUM_MODES, OUTPUT_T, 4)
    assert pi.shape == (2, NUM_MODES)


def test_inference_prediction_contract():
    torch.manual_seed(0)
    model = Diffusion_Planner(_config()).eval()
    inputs = _inputs()

    with torch.no_grad():
        _, outputs = model(inputs)

    B = 2
    P = 1 + MAX_NUM_NEIGHBORS
    assert outputs["prediction"].shape == (B, P, OUTPUT_T, POSE_DIM)
    assert outputs["trajectory"].shape == (B, NUM_MODES, OUTPUT_T, POSE_DIM)
    assert outputs["probability"].shape == (B, NUM_MODES)
    assert outputs["turn_indicator_logit"].shape == (B, TURN_INDICATOR_OUTPUT_DIM)
    assert outputs["independent_turn_indicator_logit"].shape == (B, TURN_INDICATOR_OUTPUT_DIM)
    assert torch.isfinite(outputs["prediction"]).all()

    # ego row of "prediction" must be the argmax mode, denormalized
    norm = model.decoder._state_normalizer
    best = outputs["probability"].argmax(dim=-1)
    expected = outputs["trajectory"][torch.arange(B), best] * norm.std[0] + norm.mean[0]
    assert torch.allclose(outputs["prediction"][:, 0], expected)


def test_training_loss_keys_and_backward():
    torch.manual_seed(0)
    config = _config()
    model = Diffusion_Planner(config).train()
    inputs = _inputs()

    B, Pn = 2, MAX_NUM_NEIGHBORS
    ego_future = torch.randn(B, OUTPUT_T, POSE_DIM)
    neighbors_future = torch.randn(B, Pn, OUTPUT_T, POSE_DIM)
    mask = torch.zeros(B, Pn, OUTPUT_T, dtype=torch.bool)
    mask[:, Pn // 2 :] = True  # half of the neighbors invalid

    loss = compute_plantf_training_loss(
        model, inputs, (ego_future, neighbors_future, mask), _loss_args(config)
    )

    for key in (
        "ego_planning_loss",
        "neighbor_prediction_loss",
        "mode_cls_loss",
        "turn_indicator_loss",
        "independent_turn_indicator_loss",
        "road_border_loss",
        "neighbor_collision_loss",
        "turn_indicator_accuracy",
    ):
        assert key in loss, key
        assert torch.isfinite(loss[key]), key

    total = loss["ego_planning_loss"] + loss["neighbor_prediction_loss"] + loss["mode_cls_loss"]
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_winner_takes_all_picks_closest_mode():
    """With a stubbed decoder output whose mode 1 equals GT, the WTA loss must
    select mode 1: near-zero ego loss and CE targeting mode 1."""
    torch.manual_seed(0)
    config = _config()
    args = _loss_args(config)

    B, Pn, K = 2, MAX_NUM_NEIGHBORS, NUM_MODES
    inputs = _inputs(B)
    ego_future = torch.randn(B, OUTPUT_T, POSE_DIM)
    neighbors_future = torch.zeros(B, Pn, OUTPUT_T, POSE_DIM)
    mask = torch.ones(B, Pn, OUTPUT_T, dtype=torch.bool)  # no valid neighbors

    ego_gt_norm = args.state_normalizer(torch.cat([ego_future[:, None], neighbors_future], dim=1))[
        :, 0
    ]

    trajectory = torch.randn(B, K, OUTPUT_T, POSE_DIM)
    trajectory[:, 1] = ego_gt_norm  # mode 1 is exact

    class StubModel:
        def __call__(self, merged_inputs):
            return None, {
                "trajectory": trajectory,
                "probability": torch.zeros(B, K, requires_grad=True),
                "neighbor_prediction": torch.zeros(B, Pn, OUTPUT_T, POSE_DIM),
                "turn_indicator_logit": torch.zeros(B, TURN_INDICATOR_OUTPUT_DIM),
                "independent_turn_indicator_logit": torch.zeros(B, TURN_INDICATOR_OUTPUT_DIM),
            }

    loss = compute_plantf_training_loss(
        StubModel(), inputs, (ego_future, neighbors_future, mask), args
    )

    assert loss["ego_planning_loss"] < 1e-5
    # uniform logits over K modes -> CE == log(K) regardless of the target,
    # so verify the target itself via a peaked-logit check instead
    assert loss["neighbor_prediction_loss"] == 0.0

    peaked = torch.full((B, K), -10.0)
    peaked[:, 1] = 10.0

    class PeakedStub(StubModel):
        def __call__(self, merged_inputs):
            _, out = super().__call__(merged_inputs)
            out["probability"] = peaked
            return None, out

    loss_peaked = compute_plantf_training_loss(
        PeakedStub(), inputs, (ego_future, neighbors_future, mask), args
    )
    assert loss_peaked["mode_cls_loss"] < 1e-5


def test_forward_deploy_matches_eval_forward():
    """The ONNX deploy path must produce the same prediction / probability /
    turn-indicator logit as the regular eval forward."""
    torch.manual_seed(0)
    model = Diffusion_Planner(_config()).eval()
    inputs = _inputs()

    with torch.no_grad():
        encoding = model.encoder(inputs)
        outputs = model.decoder(encoding, inputs)
        prediction, probability, turn_indicator_logit = model.decoder.forward_deploy(encoding)

    assert torch.allclose(prediction, outputs["prediction"])
    assert torch.allclose(probability, outputs["probability"])
    assert torch.allclose(turn_indicator_logit, outputs["turn_indicator_logit"])


def test_full_onnx_wrapper_keeps_diffusion_only_inputs(tmp_path):
    """The Autoware node feeds all 17 full-graph inputs by name (ORT and TensorRT
    both reject names absent from the graph), but the planTF head ignores
    sampled_trajectories / ego_current_state / delay and the legacy exporter
    prunes unused graph inputs. The planTF full wrapper must anchor them without
    changing the outputs."""
    onnx = pytest.importorskip("onnx")
    torch.manual_seed(0)
    model = Diffusion_Planner(_config()).eval()
    wrappers = build_wrappers(model)
    assert isinstance(wrappers.full, PlanTFFullONNXWrapper)
    assert wrappers.turn_indicator is None

    inputs = build_dummy_inputs()
    args = tuple(inputs[name] for name in FULL_INPUT_NAMES)
    with torch.no_grad():
        anchored_prediction, anchored_logit = wrappers.full(*args)
        plain_prediction, plain_logit = FullONNXWrapper(model).eval()(*args)
    assert torch.equal(anchored_prediction, plain_prediction)
    assert torch.equal(anchored_logit, plain_logit)

    full_onnx_path = tmp_path / "full.onnx"
    with onnx_export_backends(), torch.no_grad():
        export_onnx(
            wrappers.full,
            inputs,
            FULL_INPUT_NAMES,
            FULL_OUTPUT_NAMES,
            full_onnx_path,
            use_simplify=False,
            opset_version=17,
            external_data=False,
        )
    graph_inputs = {i.name for i in onnx.load(str(full_onnx_path)).graph.input}
    assert graph_inputs == set(FULL_INPUT_NAMES)


def test_velocity_representation_rejected():
    config = _config()
    config.use_velocity_representation = True
    with pytest.raises(NotImplementedError):
        Diffusion_Planner(config)


def test_mode_metrics_separate_candidate_quality_from_mode_selection():
    """A good oracle candidate with the wrong selected mode must have low
    minADE but worse top-1 ADE and zero mode-selection accuracy."""
    normalizer = StateNormalizer(
        mean=torch.zeros(1, 1, POSE_DIM),
        std=torch.full((1, 1, POSE_DIM), 2.0),
    )
    ego_future = torch.zeros(2, OUTPUT_T, POSE_DIM)
    trajectory_world = torch.zeros(2, NUM_MODES, OUTPUT_T, POSE_DIM)

    # sample 0: mode 1 is exact but the logits select mode 0, one metre away
    trajectory_world[0, 0, :, 0] = 1.0
    # sample 1: mode 2 is exact and selected
    trajectory_world[1, 0, :, 0] = 3.0
    trajectory_world[1, 1, :, 0] = 2.0
    probability = torch.tensor([[10.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    trajectory_norm = normalizer(trajectory_world)

    metrics = compute_plantf_mode_metrics(trajectory_norm, probability, ego_future, normalizer)

    assert torch.allclose(metrics["plantf_min_ade_k"], torch.zeros(2))
    assert torch.allclose(metrics["plantf_min_fde_k"], torch.zeros(2))
    assert torch.allclose(metrics["plantf_top1_ade"], torch.tensor([1.0, 0.0]))
    assert torch.allclose(metrics["plantf_top1_fde"], torch.tensor([1.0, 0.0]))
    assert torch.equal(metrics["plantf_mode_accuracy"], torch.tensor([0.0, 1.0]))
    assert torch.equal(metrics["plantf_mode_usage_0"], torch.tensor([1.0, 0.0]))
    assert torch.equal(metrics["plantf_mode_usage_2"], torch.tensor([0.0, 1.0]))
    assert torch.equal(metrics["plantf_miss_rate_2m"], torch.zeros(2))
