from dataclasses import dataclass
from typing import Optional


@dataclass
class ValidConfig:
    # --- 必須パラメータ ---
    resume_model_path: str
    args_json_path: str

    # --- 上書き・推論用パラメータ ---
    valid_set_list: Optional[str] = None
    save_predictions_dir: Optional[str] = None

    # --- 実行環境パラメータ ---
    batch_size: int = 32
    num_workers: int = 4
    pin_mem: bool = True
    device: str = "cuda"
    seed: int = 3407
    future_len: int = 80
    agent_num: int = 32
    predicted_neighbor_num: int = 32
    ddp: bool = True
    port: str = "22323"
