# GRPO Auto-Research Program

Autonomous GRPO hyperparameter search to eliminate off-road driving on problematic intersection scenes while preserving validation performance.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar22`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current `feat/auto-research`.
3. **Read the key files** for full context:
   - `program.md` — this file (your instructions)
   - `rlvr/grpo_config.py` — GRPOConfig dataclass (all tunable fields)
   - `rlvr/grpo_trainer.py` — training loop
   - `rlvr/grpo_loss.py` — loss computation (supports diffusion, direct_best, diffusion_low_t, diffusion_multistep)
   - `rlvr/run_experiment.py` — single-experiment runner (you create configs, it runs training + eval)
   - `~/GRPO_AUTORESEARCH_RESULTS.md` — previous research results and findings
4. **Verify environment**: `source .venv/bin/activate && python -c "import torch; print(torch.cuda.is_available())"`
5. **Initialize results.tsv**: Create `results.tsv` with the header row if it doesn't exist.
6. **Confirm and go**.

## Context: What We Know

**The problem**: Base model v3.0 goes off-road 7.2% on 100 problematic intersection scenes. GRPO training at the default lr=1e-5 doesn't fix it because LoRA weight updates are too small (~0.0001 magnitude).

**What works**: lr=1e-3 with LoRA rank 64 immediately reduces off-road to 0-2%. But it's unstable — validation performance collapses after 2-4 epochs. Strong KL (0.5) and many normal scenes (200+) help stability but don't fully solve it.

**Best result so far**: lr=1e-3, kl=0.5, 50 prob + 200 normal scenes achieved 0% off-road at epochs 2-4 but val degraded by epoch 4. See `~/GRPO_AUTORESEARCH_RESULTS.md` for all 10 previous experiments.

**Open questions**:
- Can we find a config that achieves <2% offroad AND val_reward >+7 sustained for 10+ epochs?
- Is LR scheduling (warmup/decay) the key to stability?
- Does LoRA rank affect the stability (try 32, 128)?
- Does grad_accum_groups matter (try 1, 2, 8)?
- Can reward weight tuning help (moderate feasibility boost, not 10x)?
- Does num_generations (8 vs 16 vs 32) affect the quality of advantages?

## Data Paths

All on external SSD at `/media/danielsanchez/2fb4af16-188c-4b7d-8ebb-4a7d0c90d207/`:

| Dataset | Path |
|---------|------|
| Base model | `xx1-best-model/v3.0/best_model.pth` |
| Prob scenes (100) | `path_lists/merged_20260216_20260224/path_list.json` (first 100) |
| Normal pool (71k) | `xx1_grpo_cleansed_data/path_list.json` |
| Validation (5.8k) | `xx1_validation_data/xx1_real_valid/path_list.json` |
| Output dir | `auto_research/` |

## The Experiment Script

Run an experiment:
```bash
source .venv/bin/activate
python rlvr/run_experiment.py --config <path_to_config.json> --name <experiment_name>
```

This script:
1. Loads the base model and applies LoRA
2. Creates a training set (prob + normal scenes mixed)
3. Runs GRPO training for the configured epochs
4. Evaluates on all 100 prob scenes AND 50 val scenes (deterministic, seeded)
5. Prints a results summary at the end
6. Saves everything to the output dir

The script prints a final summary like:
```
---
name:             <experiment_name>
prob_offroad:     0.021
prob_reward:      3.22
val_reward:       14.36
val_collision:    0.08
best_epoch:       3
duration_min:     25.3
checkpoint:       /media/.../auto_research/<timestamp>_<name>/lora_epoch_003
```

Extract key metrics: `grep "^prob_offroad:\|^val_reward:" run.log`

## Experiment Configs

Create JSON configs in `rlvr/configs/autoresearch/`. Each config specifies:

```json
{
  "loss_mode": "diffusion",
  "num_generations": 16,
  "train_epochs": 10,
  "kl_coef": 0.2,
  "learning_rate": 5e-4,
  "lora_rank": 64,
  "lora_alpha": 64,
  "lora_dropout": 0.05,
  "guidance_prob": 0.7,
  "enable_route_following": true,
  "enable_lane_keeping": true,
  "w_safety": 5.0,
  "w_progress": 2.0,
  "w_smooth": 0.5,
  "w_feasibility": 5.0,
  "w_centerline": 5.0,
  "grad_accum_groups": 4,
  "n_prob_scenes": 50,
  "n_normal_scenes": 100
}
```

`n_prob_scenes` and `n_normal_scenes` are custom fields consumed by `run_experiment.py`, not part of GRPOConfig.

## Logging Results

Log every experiment to `results.tsv` (tab-separated). Header:

```
name	prob_offroad	prob_reward	val_reward	val_collision	best_epoch	lr	kl	scenes	status	description
```

- status: `keep` (improved on best), `discard` (no improvement), `crash` (failed)
- Use 0.000 for crashes

Example:
```
name	prob_offroad	prob_reward	val_reward	val_collision	best_epoch	lr	kl	scenes	status	description
baseline	0.072	-2.36	8.57	0.00	0	0	0	0	keep	base model v3.0
exp001	0.021	3.22	14.36	0.08	3	1e-3	0.1	50p+50n	keep	lr=1e-3 kl=0.1
exp002	0.029	1.42	6.72	0.06	2	1e-3	0.5	50p+200n	discard	lr=1e-3 kl=0.5 more normal scenes
```

## The Experiment Loop

LOOP FOREVER:

1. **Analyze results.tsv**: Look at what's been tried, what worked, what didn't. Identify the current best configuration and the most promising direction to explore next.
2. **Design next experiment**: Create a new JSON config in `rlvr/configs/autoresearch/`. Change ONE or TWO variables from the current best (or try something new if stuck). Write a brief hypothesis for why this might help.
3. **git commit** the config file.
4. **Run**: `source .venv/bin/activate && python rlvr/run_experiment.py --config rlvr/configs/autoresearch/<config>.json --name <name> > run.log 2>&1`
5. **Check results**: `grep "^prob_offroad:\|^val_reward:\|^val_collision:\|^best_epoch:" run.log`. If grep is empty, the run crashed — `tail -50 run.log` to diagnose.
6. **Log** the result in `results.tsv`.
7. **Keep or discard**:
   - Keep if prob_offroad improved AND val_reward > +5 (acceptable)
   - Keep if val_reward improved significantly AND prob_offroad didn't get worse
   - Discard otherwise
   - Note: "keep/discard" is just for the log — we don't reset git since configs are independent
8. **Reflect**: After every 5 experiments, write a brief analysis comment in results.tsv about what patterns you see and what to try next.

## What You CAN Do

- Create new JSON configs in `rlvr/configs/autoresearch/`
- Modify `rlvr/run_experiment.py` if you need to add features (e.g., LR scheduling, new eval metrics)
- Modify `rlvr/grpo_config.py` to add new config fields
- Modify `rlvr/grpo_loss.py` or `rlvr/grpo_trainer.py` if you have an idea for a structural improvement
- Read any file in the repo for context

## What You CANNOT Do

- Modify the reward function (`rlvr/reward.py`) — it's the ground truth metric
- Modify the base model architecture (`diffusion_planner/`)
- Modify the evaluation logic in `run_experiment.py`'s `evaluate_checkpoint()` function
- Delete experiment output directories on the SSD

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas:
- Re-read previous results for patterns
- Try combining two changes that individually helped
- Try the opposite of something that failed (maybe the direction was right but magnitude wrong)
- Try structural changes to the training code
- Read the GRPO/DPO literature for ideas

The loop runs until the human interrupts you, period.

## Suggested Exploration Order

Start with the most promising directions based on previous findings:

1. **LR scheduling**: lr=1e-3 with cosine decay to 1e-4 over training (this might solve the instability)
2. **lr=5e-4 with kl=0.3**: split the difference between 1e-3 (unstable) and 3e-4 (stable but slow)
3. **More normal scenes**: lr=1e-3, kl=0.2, 50p+300n
4. **LoRA rank 128**: more capacity might help
5. **grad_accum=8**: larger effective batch for smoother gradients
6. **num_generations=32**: better advantage estimates
7. **Moderate reward boost**: w_feasibility=7, w_centerline=7, w_progress=1.5
