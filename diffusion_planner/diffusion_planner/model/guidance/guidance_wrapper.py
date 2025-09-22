import torch
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from diffusion_planner.model.guidance.collision import collision_guidance_fn

N = 1
sde = VPSDE_linear()


class GuidanceWrapper:
    def __init__(self):
        self._guidance_fns = [collision_guidance_fn]

    def __call__(self, x_in, t_input, cond, *args, **kwargs):
        """
        This function is a wrapper for the guidance functions in the model.
        """
        energy = 0

        state_normalizer = kwargs["state_normalizer"]
        observation_normalizer = kwargs["observation_normalizer"]

        B, P, _ = x_in.shape
        model = kwargs["model"]
        model_condition = kwargs["model_condition"]

        x_fix = model(x_in, t_input, **model_condition).detach() - x_in.detach()
        x_fix = x_fix.reshape(B, P, -1, 4)
        x_fix[:, :, 0] = 0.0
        x_in = x_in + x_fix.reshape(B, P, -1)

        # x_in = torch.repeat_interleave(x_in, N, dim=0) # [B * N, P, T, 4]
        # t_input = torch.repeat_interleave(t_input, N, dim=0) # [B * N]
        # kwargs["inputs"] = {k: torch.repeat_interleave(v, N, dim=0) for k, v in kwargs["inputs"].items()}

        # sigma_t = sde.marginal_prob_std(t_input)
        # sigma_t = sigma_t / torch.sqrt(1 + sigma_t ** 2)
        # x_in = torch.cat([x_in[:, :1] + sigma_t[:, None, None] * torch.randn_like(x_in[:, :1]), x_in[:, 1:]], dim=1)

        x_in = state_normalizer.inverse(x_in.reshape(B, P, -1, 4))
        kwargs["inputs"] = observation_normalizer.inverse(kwargs["inputs"])

        for guidance_fn in self._guidance_fns:
            energy += guidance_fn(x_in, t_input, cond, **kwargs)
        # energy1 = self._guidance_fns[0](x_in, t_input, cond, **kwargs)
        # energy2 = self._guidance_fns[1](x_in, t_input, cond, **kwargs)

        # energy = energy1 if energy2 < 1 else energy2

        assert not torch.isnan(energy).any()

        return energy
