from copy import copy

import torch

from diffusion_planner.utils.train_utils import openjson


class StateNormalizer:
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)

    def _mean_std_on(self, device):
        # Avoid repeating host-to-device copies for every batch.  Lazy creation
        # keeps old serialized normalizers compatible.
        cache = getattr(self, "_device_cache", None)
        if cache is None:
            cache = {}
            self._device_cache = cache
        key = str(device)
        if key not in cache:
            cache[key] = (self.mean.to(device), self.std.to(device))
        return cache[key]

    @classmethod
    def from_json(cls, args):
        data = openjson(args.normalization_file_path)
        mean = [[data["ego"]["mean"]]] + [[data["neighbor"]["mean"]]] * args.predicted_neighbor_num
        std = [[data["ego"]["std"]]] + [[data["neighbor"]["std"]]] * args.predicted_neighbor_num
        return cls(mean, std)

    def __call__(self, data):
        mean, std = self._mean_std_on(data.device)
        return (data - mean) / std

    def inverse(self, data):
        mean, std = self._mean_std_on(data.device)
        return data * std + mean

    def to_dict(self):
        return {
            "mean": self.mean.detach().cpu().numpy().tolist(),
            "std": self.std.detach().cpu().numpy().tolist(),
        }


class ObservationNormalizer:
    def __init__(self, normalization_dict):
        self._normalization_dict = normalization_dict

    @classmethod
    def from_json(cls, args):
        if isinstance(args, str):
            path = args
        else:
            path = args.normalization_file_path

        data = openjson(path)
        ndt = {}
        for k, v in data.items():
            if k not in ["ego", "neighbor"]:
                ndt[k] = {
                    "mean": torch.tensor(v["mean"], dtype=torch.float32),
                    "std": torch.tensor(v["std"], dtype=torch.float32),
                }
        return cls(ndt)

    def __call__(self, data):
        norm_data = copy(data)
        for k in self._normalization_dict:
            if k not in data:  # Check if key `k` exists in `data`
                continue
            mean, std = self._stats_on(k, data[k].device)
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
            norm_data[k] = (data[k] - mean) / std
            norm_data[k][mask] = 0
        return norm_data

    def inverse(self, data):
        norm_data = copy(data)
        for k in self._normalization_dict:
            if k not in data:  # Check if key `k` exists in `data`
                continue
            mean, std = self._stats_on(k, data[k].device)
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
            norm_data[k] = data[k] * std + mean
            norm_data[k][mask] = 0
        return norm_data

    def _stats_on(self, key, device):
        # Lazy per-device cache; this object is serialized through to_dict(), so
        # cached tensors never enter args.json or checkpoints.
        cache = getattr(self, "_device_cache", None)
        if cache is None:
            cache = {}
            self._device_cache = cache
        cache_key = (key, str(device))
        if cache_key not in cache:
            entry = self._normalization_dict[key]
            cache[cache_key] = (entry["mean"].to(device), entry["std"].to(device))
        return cache[cache_key]

    def to_dict(self):
        return {
            k: {kk: vv.detach().cpu().numpy().tolist() for kk, vv in v.items()}
            for k, v in self._normalization_dict.items()
        }
