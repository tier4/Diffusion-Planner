from functools import partial

import torch
import torch.nn as nn
from timm.models.layers import Mlp

from diffusion_planner.dimensions import TURN_INDICATOR_OUTPUT_DIM
from diffusion_planner.model.diffusion_utils.sampling import dpm_sampler
from diffusion_planner.model.diffusion_utils.sde import SDE, VPSDE_linear
from diffusion_planner.model.flow_matching_utils.ode_solver import (
    euler_integration,
    heun_integration,
    rk4_integration,
)
from diffusion_planner.model.module.dit import DiTBlock, FinalLayer, TimestepEmbedder
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len
        self._sde = VPSDE_linear()

        self.dit = DiT(
            sde=self._sde,
            depth=config.decoder_depth,
            output_dim=(config.future_len + 1) * 4,  # x, y, cos, sin
            hidden_dim=config.hidden_dim,
            heads=config.num_heads,
            dropout=dpr,
            model_type=config.diffusion_model_type,
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

    @property
    def sde(self):
        return self._sde

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

                    [training-only] "sampled_trajectories": sampled current-future ego & neighbor states,        [B, P, 1 + self._future_len, 4]
                    [training-only] "diffusion_time": timestep of diffusion process $t \in [0, 1]$,              [B]
                    ...
                }

        Returns:
            decoder_outputs: Dict
                {
                    ...
                    [training-only] "score": Predicted future states, [B, P, 1 + self._future_len, 4]
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


class DiT(nn.Module):
    def __init__(
        self,
        sde: SDE,
        depth,
        output_dim,
        hidden_dim=192,
        heads=6,
        dropout=0.1,
        mlp_ratio=4.0,
        model_type="x_start",
    ):
        super().__init__()

        assert model_type in ["score", "x_start", "flow_matching"], (
            f"Unknown model type: {model_type}"
        )
        self._model_type = model_type
        self.agent_embedding = nn.Embedding(2, hidden_dim)
        self.preproj = Mlp(
            in_features=output_dim,
            hidden_features=512,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for i in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_dim, output_dim)
        self._sde = sde
        self.marginal_prob_std = self._sde.marginal_prob_std

    @property
    def model_type(self):
        return self._model_type

    def forward(self, x, t, cross_c, neighbor_current_mask):
        """
        Forward pass of DiT.
        x: (B, P, output_dim)   -> Embedded out of DiT
        t: (B,)
        cross_c: (B, N, D)      -> Cross-Attention context
        """
        B, P, _ = x.shape

        x = self.preproj(x)

        x_embedding = torch.cat(
            [
                self.agent_embedding.weight[0][None, :],
                self.agent_embedding.weight[1][None, :].expand(P - 1, -1),
            ],
            dim=0,
        )  # (P, D)
        x_embedding = x_embedding[None, :, :].expand(B, -1, -1)  # (B, P, D)
        x = x + x_embedding

        y = self.t_embedder(t)

        attn_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)
        attn_mask[:, 1:] = neighbor_current_mask

        for block in self.blocks:
            x = block(x, cross_c, y, attn_mask)

        x = self.final_layer(x, y)

        if self._model_type == "score":
            return x / (self.marginal_prob_std(t)[:, None, None] + 1e-6)
        elif self._model_type == "x_start":
            return x
        elif self._model_type == "flow_matching":
            return x
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")
