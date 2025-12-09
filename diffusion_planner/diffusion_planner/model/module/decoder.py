from functools import partial
from typing import Dict

import torch
import torch.nn as nn

import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm
from diffusion_planner.dimensions import TURN_INDICATOR_OUTPUT_DIM
from diffusion_planner.model.flow_matching_utils.ode_solver import (
    euler_integration,
    heun_integration,
    rk4_integration,
)
from diffusion_planner.model.module.dit import DiT
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer


@torch.no_grad()
def dpm_sampler(
    model: torch.nn.Module,
    model_type: str,
    x_T: torch.Tensor,
    other_model_params: Dict,
    model_wrapper_params: Dict,
    dpm_solver_params: Dict,
):
    noise_schedule = dpm.NoiseScheduleVP()

    model_fn = dpm.model_wrapper(
        model,
        noise_schedule,
        model_type=model_type,
        model_kwargs=other_model_params,
        **model_wrapper_params,
    )

    dpm_solver = dpm.DPM_Solver(model_fn, noise_schedule, **dpm_solver_params)

    sample_dpm = dpm_solver.sample(x_T, steps=10, skip_type="logSNR")

    return sample_dpm


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len

        self.dit = DiT(
            depth=config.decoder_depth,
            output_dim=(config.future_len + 1) * 4,  # x, y, cos, sin
            hidden_dim=config.hidden_dim,
            heads=config.num_heads,
            dropout=dpr,
        )
        self.turn_indicator_predictor = nn.Linear(
            2 * (self._future_len // 10) + config.hidden_dim, TURN_INDICATOR_OUTPUT_DIM
        )

        self._state_normalizer: StateNormalizer = config.state_normalizer
        self._observation_normalizer: ObservationNormalizer = config.observation_normalizer

        # self._guidance_fn = config.guidance_fn
        self._guidance_fn = (
            config.guidance_fn if config.__dict__.get("guidance_fn") is not None else None
        )
        self._model_type = config.diffusion_model_type

    def forward(self, encoding, inputs):
        """
        Diffusion decoder process.

        Args:
            encoding: torch.Tensor
            inputs: Dict
                {
                    ...
                    "ego_current_state": current ego states,
                    "neighbor_agent_past": past and current neighbor states,

                    "sampled_trajectories": sampled current-future ego & neighbor states,        [B, P, 1 + self._future_len, 4]
                    [training-only] "diffusion_time": timestep of diffusion process $t \in [0, 1]$,              [B]
                    ...
                }

        Returns:
            decoder_outputs: Dict
                {
                    ...
                    [training-only] "model_output": Predicted future states, [B, P, 1 + self._future_len, 4]
                    [inference-only] "prediction": Predicted future states, [B, P, self._future_len, 4]
                    ...
                }

        """
        # Extract ego & neighbor current states
        ego_current = inputs["ego_current_state"][:, None, :4]
        neighbors_current = inputs["neighbor_agents_past"][
            :, : self._predicted_neighbor_num, -1, :4
        ]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        inputs["neighbor_current_mask"] = neighbor_current_mask

        current_states = torch.cat([ego_current, neighbors_current], dim=1)  # [B, P, 4]

        B, P, _ = current_states.shape
        assert P == (1 + self._predicted_neighbor_num)

        # Pool encoding to get a fixed-size representation
        encoding_pooled = torch.mean(encoding, dim=1)  # [B, D]

        sampled_trajectories = inputs["sampled_trajectories"].reshape(
            B, P, (1 + self._future_len) * 4
        )
        if self.training:
            diffusion_time = inputs["diffusion_time"]

            gt_trajectories = inputs["gt_trajectories"].reshape(B, P, (1 + self._future_len), 4)
            ego_trajectory = gt_trajectories[:, 0, 1::10, :2].reshape(
                B, 2 * (self._future_len // 10)
            )
            turn_indicator_input = torch.cat([ego_trajectory, encoding_pooled], dim=-1)
            turn_indicator_logit = self.turn_indicator_predictor(turn_indicator_input)

            return {
                "model_output": self.dit(
                    sampled_trajectories,
                    diffusion_time,
                    encoding,
                    neighbor_current_mask,
                ).reshape(B, P, -1, 4),
                "turn_indicator_logit": turn_indicator_logit,
            }
        else:
            if self._model_type == "flow_matching":
                # [B, 1 + predicted_neighbor_num, (1 + self._future_len) * 4]
                x = sampled_trajectories
                NUM_STEP = 10
                func = partial(
                    self.dit,
                    cross_c=encoding,
                    neighbor_current_mask=neighbor_current_mask,
                )
                x = euler_integration(func, x, NUM_STEP)
                # x = heun_integration(func, x, NUM_STEP)
                # x = rk4_integration(func, x, NUM_STEP)
                x = x.reshape(B, P, (1 + self._future_len) * 4)
                turn_indicator_input = torch.cat(
                    [
                        x[:, 0, 1::10, :2].reshape(B, 2 * (self._future_len // 10)),
                        encoding_pooled,
                    ],
                    dim=-1,
                )
                turn_indicator_logit = self.turn_indicator_predictor(turn_indicator_input)
                x = self._state_normalizer.inverse(x.reshape(B, P, -1, 4))[:, :, 1:]
                return {"prediction": x, "turn_indicator_logit": turn_indicator_logit}

            # [B, 1 + predicted_neighbor_num, (1 + self._future_len) * 4]
            xT = sampled_trajectories

            def initial_state_constraint(xt, t, step):
                xt = xt.reshape(B, P, -1, 4)
                xt[:, :, 0, :] = current_states
                return xt.reshape(B, P, -1)

            x0 = dpm_sampler(
                self.dit,
                self._model_type,
                xT,
                other_model_params={
                    "cross_c": encoding,
                    "neighbor_current_mask": neighbor_current_mask,
                },
                dpm_solver_params={
                    "correcting_xt_fn": initial_state_constraint,
                },
                model_wrapper_params={
                    "classifier_fn": self._guidance_fn,
                    "classifier_kwargs": {
                        "model": self.dit,
                        "model_condition": {
                            "cross_c": encoding,
                            "neighbor_current_mask": neighbor_current_mask,
                        },
                        "inputs": inputs,
                        "observation_normalizer": self._observation_normalizer,
                        "state_normalizer": self._state_normalizer,
                    },
                    "guidance_scale": 0.5,
                    "guidance_type": "classifier" if self._guidance_fn is not None else "uncond",
                },
            )
            x0 = x0.reshape(B, P, (1 + self._future_len) * 4)
            x = x0.reshape(B, P, (1 + self._future_len), 4)
            x = x[:, 0, 1::10, :2].reshape(B, 2 * (self._future_len // 10))
            turn_indicator_input = torch.cat([x, encoding_pooled], dim=-1)
            turn_indicator_logit = self.turn_indicator_predictor(turn_indicator_input)
            x0 = self._state_normalizer.inverse(x0.reshape(B, P, -1, 4))[:, :, 1:]

            return {"prediction": x0, "turn_indicator_logit": turn_indicator_logit}
