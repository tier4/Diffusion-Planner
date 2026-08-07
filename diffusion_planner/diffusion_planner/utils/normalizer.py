from copy import copy

import torch

from diffusion_planner.utils.train_utils import openjson

# Keys that ObservationNormalizer must not mean/std normalize.
#   ego / neighbor : handled by StateNormalizer instead
#   goal_pose      : handled by GoalPoseEncoder based on the goal distance
#                    (>= 100m -> learnable "goal is still far" token, otherwise divided by 100)
UNNORMALIZED_KEYS = ("ego", "neighbor", "goal_pose")


class StateNormalizer:
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)

    @classmethod
    def from_json(cls, args):
        data = openjson(args.normalization_file_path)
        mean = [[data["ego"]["mean"]]] + [[data["neighbor"]["mean"]]] * args.predicted_neighbor_num
        std = [[data["ego"]["std"]]] + [[data["neighbor"]["std"]]] * args.predicted_neighbor_num
        return cls(mean, std)

    def __call__(self, data):
        return (data - self.mean.to(data.device)) / self.std.to(data.device)

    def inverse(self, data):
        return data * self.std.to(data.device) + self.mean.to(data.device)

    def to_dict(self):
        return {
            "mean": self.mean.detach().cpu().numpy().tolist(),
            "std": self.std.detach().cpu().numpy().tolist(),
        }


def _check_last_dim(key, tensor, stats):
    """Fail with the offending key instead of an opaque broadcast error."""
    expected = stats["mean"].shape[-1]
    actual = tensor.shape[-1]
    if actual != expected:
        raise ValueError(
            f"normalization stats for '{key}' have {expected} columns "
            f"but the data has {actual}; regenerate normalization.json "
            f"(util_scripts/compute_normalization.py)"
        )


class ObservationNormalizer:
    def __init__(self, normalization_dict):
        # Drop them here so that loading an old config that still contains goal_pose
        # does not normalize it
        self._normalization_dict = {
            k: v for k, v in normalization_dict.items() if k not in UNNORMALIZED_KEYS
        }

    @classmethod
    def from_json(cls, args):
        if isinstance(args, str):
            path = args
        else:
            path = args.normalization_file_path

        data = openjson(path)
        ndt = {}
        for k, v in data.items():
            if k not in UNNORMALIZED_KEYS:
                ndt[k] = {
                    "mean": torch.tensor(v["mean"], dtype=torch.float32),
                    "std": torch.tensor(v["std"], dtype=torch.float32),
                }
        return cls(ndt)

    def __call__(self, data):
        norm_data = copy(data)
        for k, v in self._normalization_dict.items():
            if k not in data:  # Check if key `k` exists in `data`
                continue
            _check_last_dim(k, data[k], v)
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
            norm_data[k] = (data[k] - v["mean"].to(data[k].device)) / v["std"].to(data[k].device)
            norm_data[k][mask] = 0
        return norm_data

    def inverse(self, data):
        norm_data = copy(data)
        for k, v in self._normalization_dict.items():
            if k not in data:  # Check if key `k` exists in `data`
                continue
            _check_last_dim(k, data[k], v)
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
            norm_data[k] = data[k] * v["std"].to(data[k].device) + v["mean"].to(data[k].device)
            norm_data[k][mask] = 0
        return norm_data

    def to_dict(self):
        return {
            k: {kk: vv.detach().cpu().numpy().tolist() for kk, vv in v.items()}
            for k, v in self._normalization_dict.items()
        }
