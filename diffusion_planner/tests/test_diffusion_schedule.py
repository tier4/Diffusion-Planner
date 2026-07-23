import torch
from diffusion_planner.model.diffusion_utils.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear


def test_vpsde_alpha_matches_marginal_mean():
    sde = VPSDE_linear()
    t = torch.linspace(1e-3, 1.0, 6).reshape(2, 1, 3, 1)
    x = torch.randn(2, 4, 3, 4)

    mean, std = sde.marginal_prob(x, t)

    torch.testing.assert_close(mean, sde.marginal_alpha(t) * x)
    torch.testing.assert_close(std, sde.marginal_prob_std(t), rtol=1e-6, atol=1e-7)


def test_dpm_timesteps_match_the_previous_cpu_construction():
    schedule = NoiseScheduleVP()
    solver = DPM_Solver(lambda x, t: x, schedule)
    t_T, t_0, steps = 1.0, 1e-3, 6

    for skip_type in ("logSNR", "time_uniform", "time_quadratic"):
        actual = solver.get_time_steps(skip_type, t_T, t_0, steps, torch.device("cpu"))
        if skip_type == "logSNR":
            lambda_T = schedule.marginal_lambda(torch.tensor(t_T))
            lambda_0 = schedule.marginal_lambda(torch.tensor(t_0))
            log_snr = torch.linspace(lambda_T.item(), lambda_0.item(), steps + 1)
            expected = schedule.inverse_lambda(log_snr)
        elif skip_type == "time_uniform":
            expected = torch.linspace(t_T, t_0, steps + 1)
        else:
            expected = torch.linspace(t_T**0.5, t_0**0.5, steps + 1).pow(2)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
