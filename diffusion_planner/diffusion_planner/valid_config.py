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
    override_open_loop_list: Optional[str] = None
    override_only: bool = False

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

    enable_temporal_stability_eval: bool = True
    enable_replan_consistency_eval: bool = True
    replan_consistency_expected_gap: int = 1
    enable_epdms_eval: bool = True
    enable_pdms_eval: bool = False
    epdms_eval_use_agent_boxes: bool = True
    epdms_eval_use_road_border: bool = True
