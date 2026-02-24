import torch
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from diffusion_planner.model.guidance.centerline_following import centerline_following_fn
from diffusion_planner.model.guidance.collision import collision_guidance_fn
from diffusion_planner.model.guidance.lane_keeping import lane_keeping_fn
from diffusion_planner.model.guidance.route_following import route_following_fn

N = 1
sde = VPSDE_linear()


class GuidanceWrapper:
    def __init__(
        self,
        use_collision: bool = True,
        use_route_following: bool = False,
        use_lane_keeping: bool = False,
        use_centerline_following: bool = False,
    ):
        """Accumulates energy from one or more guidance functions.

        Args:
            use_collision: Enable collision-avoidance guidance.
            use_route_following: Enable route-following guidance.
            use_lane_keeping: Enable lane-keeping guidance.
        """
        self._guidance_fns = []
        if use_collision:
            self._guidance_fns.append(collision_guidance_fn)
        if use_route_following:
            self._guidance_fns.append(route_following_fn)
        if use_lane_keeping:
            self._guidance_fns.append(lane_keeping_fn)
        if use_centerline_following:
            self._guidance_fns.append(centerline_following_fn)

        if not self._guidance_fns:
            raise ValueError(
                "GuidanceWrapper requires at least one guidance function. "
                "Set use_collision, use_route_following, or use_lane_keeping to True."
            )

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

        x_in = state_normalizer.inverse(x_in.reshape(B, P, -1, 4))
        kwargs["inputs"] = observation_normalizer.inverse(kwargs["inputs"])

        for guidance_fn in self._guidance_fns:
            e = guidance_fn(x_in, t_input, cond, **kwargs)
            if torch.isnan(e).any():
                print(f"Warning: NaN energy from {guidance_fn.__name__}, skipping")
                continue
            energy += e

        if isinstance(energy, int):
            # No guidance function produced valid energy
            energy = torch.zeros(x_in.shape[0], device=x_in.device)

        return energy
