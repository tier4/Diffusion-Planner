"""Train the trajectory predictor.

Launched under torch.distributed.run, normally via train_run.py:

    python train_run.py --exp_name my_exp \
        --train_set_list /path/to/train_list.json \
        --valid_set_list /path/to/valid_list.json

All flags are declared on :class:`diffusion_planner.config.TrainConfig` with
``cli(...)`` and mirrored on train_run.py.
"""

from diffusion_planner.config import TrainConfig, build_config, build_parser
from diffusion_planner.train import model_training
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer


def main() -> None:
    args = build_parser(TrainConfig, description=__doc__).parse_args()
    cfg = build_config(TrainConfig, args)
    cfg.state_normalizer = StateNormalizer.from_json(cfg)
    cfg.observation_normalizer = ObservationNormalizer.from_json(cfg)
    model_training(cfg)


if __name__ == "__main__":
    main()
