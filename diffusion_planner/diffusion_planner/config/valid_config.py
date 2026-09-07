from dataclasses import dataclass

from .base_config import BaseConfig


@dataclass
class ValidConfig(BaseConfig):
    resume_model_path: str = ""
    args_json_path: str = ""

    batch_size: int = 32
    num_workers: int = 4
    pin_mem: bool = True
    future_len: int = 80
    agent_num: int = 32
    predicted_neighbor_num: int = 32

    valid_set_list: str | None = None
    save_predictions_dir: str | None = None
    scenario_based_open_loop_list: str | None = None
    scenario_based_open_loop_only: bool = False
