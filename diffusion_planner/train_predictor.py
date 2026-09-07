"""Train the trajectory predictor.

Launched under torch.distributed.run, normally via train_run.py:

    python train_run.py --exp_name my_exp \
        --train_set_list /path/to/train_list.json \
        --valid_set_list /path/to/valid_list.json

All flags are declared on :class:`diffusion_planner.config.TrainConfig` with
``cli(...)`` and mirrored on train_run.py.
"""

import dataclasses
import json

from diffusion_planner.config import TrainConfig, build_config, build_parser
from diffusion_planner.train import model_training
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer


def apply_overrides_json(cfg: TrainConfig, overrides_path: str) -> TrainConfig:
    """Apply a JSON object of field overrides onto a built config.

    Unknown keys fail loudly: a typo or a field renamed upstream must not
    silently fall back to the default (the caller believes it set the value).
    """
    if not overrides_path:
        return cfg
    with open(overrides_path) as f:
        overrides = json.load(f)
    if not isinstance(overrides, dict):
        raise ValueError(f"{overrides_path} must contain a JSON object")
    all_fields = {f.name: f for f in dataclasses.fields(TrainConfig)}
    unknown = sorted(set(overrides) - set(all_fields))
    if unknown:
        raise ValueError(f"{overrides_path} sets keys that are not TrainConfig fields: {unknown}")
    cli_exposed = sorted(k for k in overrides if all_fields[k].metadata.get("cli"))
    if cli_exposed:
        raise ValueError(
            f"{overrides_path} sets CLI-exposed fields {cli_exposed}; pass those as "
            "command-line flags — a file value would silently shadow the flag"
        )
    # JSON cannot express every annotation, but the scalar cases cover the
    # dangerous silent coercions ('false' is a truthy str; 4.0 is not an int).
    scalar = {"bool": bool, "int": int, "float": (int, float), "str": str}
    for key, value in overrides.items():
        field_type = all_fields[key].type
        annotation = getattr(field_type, "__name__", str(field_type))
        expected = scalar.get(annotation)
        if expected is not None and not isinstance(value, expected):
            raise ValueError(
                f"{overrides_path}: {key} expects {annotation}, "
                f"got {type(value).__name__} ({value!r})"
            )
        if annotation != "bool" and isinstance(value, bool):
            raise ValueError(f"{overrides_path}: {key} expects {annotation}, got bool")
        setattr(cfg, key, value)
    return cfg


def main() -> None:
    args = build_parser(TrainConfig, description=__doc__).parse_args()
    cfg = build_config(TrainConfig, args)
    cfg = apply_overrides_json(cfg, cfg.train_overrides_json)
    cfg.state_normalizer = StateNormalizer.from_json(cfg)
    cfg.observation_normalizer = ObservationNormalizer.from_json(cfg)
    model_training(cfg)


if __name__ == "__main__":
    main()
