"""Workflow registry: declarative descriptions of the autoresearch CLI tools.

Each :class:`Workflow` declares the module / script to run, the environment it needs,
and an ordered list of :class:`ArgSpec` form fields. The panel auto-generates a form
from these specs and :func:`build_command` turns the filled form into an argv list.

Nothing here imports torch or runs inference — this module is a pure description of how
the existing CLI tools are invoked, so it stays cheap to import and easy to audit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Argument "kind" controls the Gradio widget the panel renders and how the value is
# serialised into argv. Booleans carry an extra ``bool_style`` (see ArgSpec).
KINDS = ("path", "file", "dir", "str", "int", "float", "bool", "choice")


@dataclass
class ArgSpec:
    """One CLI argument / form field."""

    name: str  # python identifier, e.g. "model_path"
    kind: str = "str"
    label: str = ""  # display label; defaults to a prettified ``name``
    flag: str | None = None  # literal CLI token; defaults to f"--{name}". None+positional.
    default: object = None
    required: bool = False
    help: str = ""
    choices: list[str] | None = None
    positional: bool = False  # positional arg (no flag), emitted in declaration order
    multi: bool = False  # nargs="+": value is whitespace-separated, each emitted as a token
    launcher_only: bool = (
        False  # consumed by the runner (e.g. torchrun port), never in the tool argv
    )
    # bool serialisation: "store_true" → emit flag when True; "value" → "--flag true/false";
    # "optional" → argparse.BooleanOptionalAction ("--flag" / "--no-flag").
    bool_style: str = "store_true"
    # If set, this field is sourced from the central asset library rather than typed per-tab.
    # One of: "models" | "datasets" | "reward_configs" | "maps" | "ego_shape" | "output_dir".
    # The app renders a dropdown over registered names and resolves the path at Run.
    shared: str | None = None
    # If set, this field is NOT rendered; its value is derived at Run from a sibling model arg's
    # selected library entry. derive_from = the sibling arg name; derive_field = entry key
    # ("args_json" | "lora_dir"). Lets a chosen model carry its args.json / LoRA automatically.
    derive_from: str | None = None
    derive_field: str | None = None
    # Not rendered; its ``default`` is always used (e.g. a fixed flag the user shouldn't fiddle).
    hidden: bool = False
    # Auto-derived output path under <workspace output_dir>/<workflow key>/<run name>. Not
    # rendered; the app shows a single "Run name" field and fills these. Values:
    #   "dir"          -> the run folder itself
    #   "dir:<sub>"    -> a subfolder inside the run folder
    #   "file:<name>"  -> a file inside the run folder
    auto: str | None = None
    # For a shared-list field: allow picking MULTIPLE library entries (a multiselect dropdown).
    # At Run the panel merges the selected datasets' scene lists into one combined JSON.
    multi_select: bool = False

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"ArgSpec {self.name!r}: unknown kind {self.kind!r}")
        if not self.label:
            self.label = self.name.replace("_", " ").strip().capitalize()
        if self.flag is None and not self.positional and not self.launcher_only:
            self.flag = f"--{self.name}"


@dataclass
class Workflow:
    """A runnable CLI tool."""

    key: str
    title: str
    args: list[ArgSpec]
    module: str | None = None  # "python -m <module>"
    script_path: str | None = None  # repo-relative script (e.g. ros_scripts/torch2onnx.py)
    env: str = "venv"  # "venv" | "ros"
    torchrun: bool = False  # launch under torchrun (DDP) instead of plain python
    server: bool = False  # long-lived interactive server (Scene Editor) vs fire-and-finish
    # What this workflow produces, for auto-placement under the workspace:
    #   "scenes" -> datasets/scenes/<name>/ (+ <name>.json list)  "routes" -> datasets/routes/<name>/
    #   None     -> runs/<key>/<name>/
    creates: str | None = None
    description: str = ""
    # Best-effort resolver: filled arg values -> dict of output locations the UI can show.
    outputs: Callable[[dict], dict] | None = None

    def spec(self, name: str) -> ArgSpec:
        for a in self.args:
            if a.name == name:
                return a
        raise KeyError(f"{self.key}: no arg {name!r}")


def _missing_required(wf: Workflow, values: dict) -> list[str]:
    out = []
    for a in wf.args:
        if not a.required:
            continue
        v = values.get(a.name)
        if v is None or (isinstance(v, str) and v.strip() == ""):
            out.append(a.name)
    return out


def build_command(wf: Workflow, values: dict) -> list[str]:
    """Turn filled form ``values`` into the tool-specific argv tail (no python prefix).

    Raises ``ValueError`` listing any blank required fields — the panel surfaces this
    rather than launching a doomed subprocess. We never substitute a silent default for
    a required arg (per the no-silent-fallbacks rule).
    """
    missing = _missing_required(wf, values)
    if missing:
        raise ValueError(f"{wf.title}: missing required field(s): {', '.join(missing)}")

    positional: list[str] = []
    optional: list[str] = []
    for a in wf.args:
        # Launcher-only meta args are consumed by the runner (e.g. torchrun master_port),
        # not passed to the tool's argv.
        if a.launcher_only:
            continue
        v = values.get(a.name, a.default)
        if a.kind == "bool":
            v = bool(v)
            if a.bool_style == "store_true":
                if v:
                    optional.append(a.flag)
            elif a.bool_style == "value":
                optional += [a.flag, "true" if v else "false"]
            elif a.bool_style == "optional":
                optional.append(a.flag if v else a.flag.replace("--", "--no-", 1))
            else:
                raise ValueError(f"{a.name}: bad bool_style {a.bool_style!r}")
            continue

        # Skip unset optionals (blank string / None). Required blanks were caught above.
        if v is None or (isinstance(v, str) and v.strip() == ""):
            continue

        if a.positional:
            positional.append(str(v))
            continue

        if a.multi:
            tokens = v if isinstance(v, (list, tuple)) else str(v).split()
            if tokens:
                optional.append(a.flag)
                optional += [str(t) for t in tokens]
            continue

        optional += [a.flag, str(v)]

    # Positionals first (argparse convention used by torch2onnx).
    return positional + optional


# --------------------------------------------------------------------------------------
# Shared arg builders (preset-aware) so common fields stay consistent across workflows.
# --------------------------------------------------------------------------------------
def _model_path(
    required: bool = True, name: str = "model_path", label: str = "Model (.pth)"
) -> ArgSpec:
    return ArgSpec(
        name,
        "file",
        label=label,
        shared="models",
        required=required,
        help="Registered model. args.json / lora_dir come from the chosen library entry.",
    )


def _ego_shape(required: bool = True) -> ArgSpec:
    return ArgSpec(
        "ego_shape",
        "str",
        label="Ego shape (WB,L,W)",
        shared="ego_shape",
        required=required,
        help="Wheelbase,Length,Width in metres, e.g. 4.76,7.24,2.29",
    )


def _grpo_config(name: str = "config", label: str = "GRPO / generation config") -> ArgSpec:
    return ArgSpec(
        name,
        "file",
        label=label,
        shared="grpo_configs",
        required=True,
        help="GRPOConfig JSON (configs/grpo/). Generation + training settings.",
    )


def _reward_config(name: str = "config", label: str = "Reward config JSON") -> ArgSpec:
    return ArgSpec(
        name,
        "file",
        label=label,
        shared="reward_configs",
        required=True,
        help="Registered reward/GRPO config JSON. Must set centerline_usage_mode=baselink.",
    )


def _output_dir(required: bool = True) -> ArgSpec:
    # Auto-derived: <workspace output_dir>/<workflow key>/<run name>. Not typed per-run.
    return ArgSpec("output_dir", "dir", auto="dir", required=required)


def _scenes(
    name: str = "scenes",
    label: str = "Scenes JSON",
    multi: bool = False,
    shared: str | None = "scene_datasets",
    required: bool = True,
) -> ArgSpec:
    # Scene-dataset dropdowns are multi-select (pick several datasets; merged at Run).
    return ArgSpec(
        name,
        "file",
        label=label,
        required=required,
        multi=multi,
        shared=shared,
        multi_select=(shared == "scene_datasets"),
        help="Pick one or more registered scene datasets (merged into one list at Run).",
    )


def _lora(name: str = "lora_path", label: str = "LoRA") -> ArgSpec:
    return ArgSpec(
        name,
        "dir",
        label=label,
        shared="loras",
        help="Registered LoRA adapter dir (optional). Combine freely with any base model.",
    )


def _policy(
    name: str = "policy_dir", label: str = "Guidance model", required: bool = False
) -> ArgSpec:
    """A guidance / exploration-policy field. The single place to add 'guidance model' to any
    tool — pair it after a model (+LoRA) and the form lays out model / LoRA / guidance on one row.
    """
    return ArgSpec(
        name,
        "dir",
        label=label,
        shared="policies",
        required=required,
        help="Exploration/guidance policy dir (optional). Adds learned guidance on top of the "
        "model — combine freely with any base model and LoRA.",
    )


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------
WORKFLOWS: dict[str, Workflow] = {}


def _register(wf: Workflow) -> Workflow:
    WORKFLOWS[wf.key] = wf
    return wf


# --- Train: ranked-SFT ----------------------------------------------------------------
_register(
    Workflow(
        key="train_ranked_sft",
        title="Train (RSFT / Ranked-SFT)",
        module="rlvr.autoresearch.run_experiment",
        description="Ranked-SFT (RSFT): generate K trajectories per scene, rank by reward, SFT on "
        "the winner (set ranked_sft_mode in the config). PRiSM is RSFT on perturbation-mined "
        "scenes. Writes a timestamped run dir with per-epoch LoRA + eval.",
        args=[
            _grpo_config(label="Experiment config (sets ranked_sft_mode, lr, train_epochs, …)"),
            ArgSpec("name", "str", label="Experiment name", required=True),
            _model_path(label="Baseline model (warmstart from)"),
            _scenes(name="train_scenes", label="Training scenes"),
            _scenes(name="val_scenes", label="Validation scenes"),
            ArgSpec(
                "train_epochs",
                "int",
                label="Epochs (optional override)",
                help="Leave blank to use the train_epochs set in the experiment config above.",
            ),
            ArgSpec(
                "sft_batch_size",
                "int",
                label="Batch size — scenes per forward pass (optional, speeds it up)",
                help="Config default is 1 (sequential = slow). Try 8 or 16 to batch scenes per "
                "forward pass. grad_accum_groups auto-adjusts. Lower it if you hit GPU OOM.",
            ),
            _output_dir(),
            # Always skip the in-training baseline eval (use the dedicated Evaluate tab instead).
            ArgSpec("skip_baseline", "bool", default=True, hidden=True),
        ],
        outputs=lambda v: {"run_root": v.get("output_dir")},
    )
)

# --- Eval: deterministic avoidance ----------------------------------------------------
_register(
    Workflow(
        key="eval_det_avoidance",
        title="Eval: det avoidance (sc only)",
        module="rlvr.autoresearch.tools.eval_det_avoidance",
        description="Avoidance-focused det eval (sc + rb + cl). Superseded in the UI by the unified "
        "Model evaluation tab; kept for scripts.",
        args=[
            _model_path(),
            _lora(),
            _scenes(),
            _reward_config(),
            _ego_shape(),
            _output_dir(),
            ArgSpec("batch_size", "int", default=32),
        ],
        outputs=lambda v: (
            {"summary_json": str(Path(v["output_dir"]) / "summary.json")}
            if v.get("output_dir")
            else {}
        ),
    )
)

# --- Eval: unified full metrics (deterministic) ---------------------------------------
_register(
    Workflow(
        key="eval_full_metrics",
        title="Model evaluation — full metrics",
        module="rlvr.autoresearch.tools.eval_full_metrics",
        description="One deterministic forward pass per scene scored against ALL reward metrics: "
        "avoidance (sc clearance + crossings), road border, lane, centerline, path length, "
        "collision + kinematic flags — with full distribution stats. Optional LoRA. Add a 2nd "
        "model for a head-to-head A/B table, and tick Render scenes to dump per-scene PNGs. Use "
        "an SC-enabled reward config so avoidance (sc) is actually measured. Writes summary.json.",
        args=[
            _model_path(name="model_path", label="Model A (.pth)"),
            _lora(name="lora_path", label="LoRA A"),
            _model_path(
                name="model_b", label="Model B (.pth, optional head-to-head)", required=False
            ),
            _lora(name="lora_b", label="LoRA B"),
            ArgSpec("label_a", "str", default="A"),
            ArgSpec("label_b", "str", default="B"),
            _scenes(),
            _reward_config(),
            _ego_shape(),
            _output_dir(),
            ArgSpec("render", "bool", label="Render per-scene PNGs"),
            ArgSpec("batch_size", "int", default=32),
        ],
        outputs=lambda v: (
            {"summary_json": str(Path(v["output_dir"]) / "summary.json")}
            if v.get("output_dir")
            else {}
        ),
    )
)

# --- Eval: guided avoidance (exploration policy) --------------------------------------
_register(
    Workflow(
        key="eval_policy_avoidance",
        title="Eval: guided avoidance (policy)",
        module="rlvr.autoresearch.tools.eval_policy_avoidance",
        description="Avoidance eval under a guidance/exploration policy. The guidance envelope is "
        "loaded from the policy dir — do NOT pass envelope flags (per the eval rule).",
        args=[
            _model_path(),
            _policy(name="policy_dir", label="Guidance model", required=True),
            _scenes(label="Scenes"),
            _reward_config(),
            _ego_shape(),
            _output_dir(),
            ArgSpec("render", "bool", label="Render per-scene PNGs"),
        ],
        outputs=lambda v: {"dir": v.get("output_dir")},
    )
)

# --- Eval: L2 (valid_predictor, torchrun DDP) -----------------------------------------
_register(
    Workflow(
        key="eval_l2",
        title="Eval: L2 (valid_predictor)",
        script_path="diffusion_planner/valid_predictor.py",
        torchrun=True,
        description="DDP L2 validation. The validation set's vehicle/ego-shape must match the "
        "model's training platform. Merge any LoRA into a .pth first (use the Merge+Export tab).",
        args=[
            _model_path(name="resume_model_path", label="Model (.pth)"),
            ArgSpec(
                "args_json_path",
                "file",
                label="args.json",
                required=True,
                derive_from="resume_model_path",
                derive_field="args_json",
            ),
            _scenes(name="valid_set_list", label="Validation set JSON"),
            ArgSpec("batch_size", "int", default=32),
            # valid_predictor only runs under torch DDP (even on 1 GPU); always on, hidden.
            ArgSpec("ddp", "bool", default=True, bool_style="value", hidden=True),
            ArgSpec(
                "master_port",
                "str",
                label="torchrun master_port",
                default="29505",
                launcher_only=True,
                help="Used by the torchrun launcher, not the script.",
            ),
        ],
    )
)

# --- Eval: road border ----------------------------------------------------------------
_register(
    Workflow(
        key="eval_border_distance",
        title="Eval: road border",
        module="rlvr.autoresearch.eval_border_distance",
        description="Road-border clearance distribution. Optionally visualise the worst scenes.",
        args=[
            _model_path(),
            _lora(),
            ArgSpec(
                "args_json",
                "file",
                label="args.json",
                derive_from="model_path",
                derive_field="args_json",
            ),
            _scenes(),
            ArgSpec("tag", "str", default="model"),
            ArgSpec("visualize", "bool", label="Visualize worst scenes"),
            ArgSpec("worst_n", "int", label="# worst to visualize", default=10),
            _output_dir(required=False),
        ],
    )
)

# --- Eval: detailed metrics (multi-LoRA) ----------------------------------------------
_register(
    Workflow(
        key="eval_detailed_metrics",
        title="Eval: detailed metrics",
        module="rlvr.autoresearch.tools.eval_detailed_metrics",
        description="Per-scene centerline + RB distribution percentiles and threshold buckets "
        "across one or more LoRA dirs.",
        args=[
            _model_path(),
            _scenes(),
            _reward_config(label="GRPO config JSON"),
            ArgSpec(
                "loras",
                "str",
                label="LoRA dirs (space-separated)",
                required=True,
                multi=True,
                help="One or more LoRA dirs (lora_epoch_NNN/ or lora_latest/).",
            ),
            ArgSpec("labels", "str", label="Labels (space-separated, optional)", multi=True),
            ArgSpec("batch_size", "int", default=150),
            ArgSpec("dump_json", "file", label="Dump JSON (optional)"),
        ],
    )
)

# --- Merge LoRA -----------------------------------------------------------------------
_register(
    Workflow(
        key="merge_lora",
        title="Merge LoRA → .pth",
        module="preference_optimization.merge_lora",
        description="Fuse a LoRA adapter into the base weights, producing a deployable .pth.",
        args=[
            _model_path(label="Base / warmstart model (.pth)"),
            _lora(name="lora_dir", label="LoRA to merge"),
            ArgSpec("output", "file", auto="file:merged.pth"),
        ],
        outputs=lambda v: {"merged": v.get("output")},
    )
)

# --- ONNX export ----------------------------------------------------------------------
_register(
    Workflow(
        key="torch2onnx",
        title="Export ONNX",
        script_path="ros_scripts/torch2onnx.py",
        description="Export a model to ONNX (written into the model's own dir). Just pick the "
        "model — its folder (best_model.pth + args.json) is the deploy dir.",
        args=[
            # UI-only selector: a registered model = a deploy dir. Not emitted to argv
            # (launcher_only); root_dir below is derived from its folder.
            ArgSpec(
                "model",
                "file",
                label="Model to export",
                shared="models",
                required=True,
                launcher_only=True,
            ),
            ArgSpec("root_dir", "dir", positional=True, derive_from="model", derive_field="dir"),
            ArgSpec(
                "use_simplify", "bool", label="Simplify graph (onnxsim)", flag="--use-simplify"
            ),
            ArgSpec("opset_version", "int", flag="--opset-version", default=20, hidden=True),
        ],
    )
)

# --- PRiSM: disturb_and_replay --------------------------------------------------------
_register(
    Workflow(
        key="disturb_and_replay",
        title="PRiSM: disturb_and_replay",
        module="rlvr.autoresearch.tools.disturb_and_replay",
        creates="scenes",
        description="Generate perturbed variants (parallel offset / yaw / jitter) of warm scenes. "
        "All output NPZ fields are in the perturbed-ego frame. Emits manifest.json.",
        args=[
            _scenes(label="Warm scenes JSON"),
            _output_dir(),
            ArgSpec("output_scene_list", "file", auto="file:scenes.json", required=True),
            ArgSpec(
                "kind",
                "choice",
                label="Kind",
                default="parallel_only",
                choices=["default", "parallel_only", "yaw_only", "combined"],
            ),
            ArgSpec("n_per_scene", "int", label="Variants per scene", default=5),
            ArgSpec("offsets", "str", label="Lateral offsets (m)", default="0.25,0.5,1.0"),
            ArgSpec("yaw_degs", "str", label="Yaw magnitudes (deg)", default="5,10,15"),
            ArgSpec(
                "reject_out_of_lane",
                "bool",
                label="Reject out-of-lane",
                default=True,
                bool_style="optional",
            ),
            ArgSpec("reject_threshold", "float", label="Reject threshold (m)", default=0.15),
            ArgSpec("seed", "int", default=0),
            _ego_shape(),
        ],
        outputs=lambda v: {"scene_list": v.get("output_scene_list"), "dir": v.get("output_dir")},
    )
)

# --- PRiSM: viz_p4_recovery -----------------------------------------------------------
_register(
    Workflow(
        key="viz_p4_recovery",
        title="PRiSM: rank candidates (K=N) → recovery score",
        module="rlvr.autoresearch.tools.viz_p4_recovery",
        description="PRiSM step 2 — for each perturbed scene, generate K candidate trajectories "
        "under the model and rank by reward; records how well rank-1 recovers (t0_cl/det_cl/top1_cl "
        "+ safety flags) into summary.json. Feeds the percentile filter, which keeps only the "
        "scenes worth RSFT-training on.",
        args=[
            _model_path(),
            _lora(),
            _scenes(label="Perturbed scenes (dataset)"),
            _grpo_config(label="GRPO / generation config (guidance)"),
            _output_dir(),
            ArgSpec("manifest", "file", label="disturb manifest.json (optional)"),
            ArgSpec("K", "int", label="K (generations)", default=8),
            ArgSpec("noise_min", "float", default=0.5),
            ArgSpec("noise_max", "float", default=2.0),
            ArgSpec(
                "scene_batch_size",
                "int",
                label="Scene batch size",
                default=8,
                help="Batched K-generation. Requires --no_viz when > 1.",
            ),
            ArgSpec("no_viz", "bool", label="No viz (summary.json only)", default=True),
            ArgSpec("max_scenes", "int", label="Max scenes (optional)"),
            ArgSpec("seed", "int", default=0),
            _ego_shape(),
        ],
        outputs=lambda v: (
            {"summary_json": str(Path(v["output_dir"]) / "summary.json")}
            if v.get("output_dir")
            else {}
        ),
    )
)

# --- PRiSM: percentile_filter_perturbed -----------------------------------------------
_register(
    Workflow(
        key="percentile_filter_perturbed",
        title="PRiSM: filter scenes",
        module="rlvr.autoresearch.tools.percentile_filter_perturbed",
        description="Keep only the scenes worth training on: the top X% by best-candidate reward, "
        "dropping scenes that no candidate improved. Writes the SFT scene list.",
        args=[
            ArgSpec(
                "summary",
                "file",
                label="Ranking summary.json (from Step 2)",
                required=True,
                help="The summary.json written by the rank step.",
            ),
            ArgSpec("output_scenes", "file", auto="file:filtered_scenes.json", required=True),
            ArgSpec("output_report", "file", auto="file:filter_report.json"),
            ArgSpec(
                "percentile",
                "float",
                label="Keep top percentile by reward (0–100)",
                default=50.0,
                help="e.g. 50 keeps the better-recovering half of the scenes.",
            ),
            ArgSpec(
                "min_top1_vs_det",
                "float",
                label="Min improvement over baseline (drops non-improving scenes)",
                default=0.0,
                help="A scene is kept only if its best candidate beats the deterministic plan by "
                "at least this much. 0 drops scenes that got worse.",
            ),
            # Advanced no-poison guards — fixed at their canonical defaults, not shown.
            ArgSpec("det_cl_max", "float", hidden=True),
            ArgSpec("top1_cl_min", "float", hidden=True),
            ArgSpec("eligible_t0_max", "float", default=0.0, hidden=True),
        ],
        outputs=lambda v: {"scene_list": v.get("output_scenes")},
    )
)

# --- Reproducer: collision miner ------------------------------------------------------
_register(
    Workflow(
        key="mine_collisions",
        title="Reproducer: mine collisions",
        module="rlvr.autoresearch.tools.mine_collisions_reproducer",
        creates="scenes",  # collision windows land in datasets/scenes/<name>/ → scannable dataset
        description="Closed-loop perception reproducer over a pre-converted NPZ corpus (map-free). "
        "Writes ranked hits JSONL; --save_dir saves pre-collision training NPZ batches one-pass.",
        args=[
            ArgSpec(
                "npz_root", "dir", label="Route corpus", shared="route_datasets", required=True
            ),
            _model_path(),
            ArgSpec("out", "file", auto="file:hits.jsonl", required=True),
            ArgSpec(
                "save_thresh",
                "float",
                label="Collision distance (m) — flag a scene when ego↔neighbor ≤ this",
                default=0.2,
            ),
            ArgSpec("seg_len", "int", label="Segment length (frames)", default=600),
            ArgSpec("batch_size", "int", label="Segment batch size", default=8),
            ArgSpec("save_dir", "dir", auto="dir:collision_batches"),
            ArgSpec("save_pre_steps", "int", label="Pre-steps to save", default=80),
            ArgSpec("save_min_ego_speed", "float", label="Min ego speed (m/s)", default=0.5),
            ArgSpec("unstick_after", "int", label="Unstick after (steps)", default=300),
            ArgSpec("dump_hits", "int", label="Render top-N hit segments to PNGs", default=0),
            ArgSpec("max_routes", "int", label="Max routes (debug)", default=-1),
            ArgSpec("max_segments", "int", label="Max segments (debug)", default=-1),
            ArgSpec("device", "str", label="Device (optional)"),
        ],
        outputs=lambda v: {"hits_jsonl": v.get("out"), "save_dir": v.get("save_dir")},
    )
)

# --- Viz: open-loop dual-model perfect-track ------------------------------------------
_register(
    Workflow(
        key="ghost_replay_openloop",
        title="Viz: open-loop ghost (A/B)",
        module="rlvr.autoresearch.tools.ghost_replay_openloop",
        description="Per-scene perfect-tracking open-loop replay: baseline vs model, history + 80-step "
        "plan, neighbor boxes. Outputs PNG seq + WebM per scene.",
        args=[
            _model_path(name="model_baseline", label="Baseline model (.pth)"),
            _lora(name="lora_baseline", label="LoRA (baseline)"),
            _policy(name="policy_baseline", label="Guidance (baseline)"),
            _model_path(name="model_best", label="Best model (.pth)", required=False),
            _lora(name="lora_best", label="LoRA (best)"),
            _policy(name="policy_best", label="Guidance (best)"),
            ArgSpec("label_baseline", "str", default="baseline"),
            ArgSpec("label_best", "str", default="best"),
            _scenes(),
            _output_dir(),
            _ego_shape(),
            ArgSpec("view_half", "float", label="View half (m)", default=28.0),
            ArgSpec("fps", "int", default=10),
            ArgSpec("hist_steps", "int", label="History steps", default=30),
        ],
        outputs=lambda v: {"dir": v.get("output_dir")},
    )
)

# --- Viz: closed-loop dual-model ghost ------------------------------------------------
_register(
    Workflow(
        key="compare_models_ghost",
        title="Viz: closed-loop ghost (A/B)",
        module="rlvr.autoresearch.tools.compare_models_ghost",
        description="Per-step closed-loop A/B comparison with both ego footprints + stopped-neighbor "
        "OBBs. Each side is model + optional LoRA + optional guidance policy — compare any "
        "combination (e.g. plain model vs model+LoRA, or model+LoRA vs model+guidance). When only a "
        "B-side guidance policy is set and Model B is left empty, side B reuses Model A. Optional WebM.",
        args=[
            _model_path(name="model_a", label="Model A (.pth)"),
            _lora(name="lora_a", label="LoRA A"),
            _policy(name="policy_a", label="Guidance A"),
            ArgSpec("label_a", "str", default="A"),
            _model_path(name="model_b", label="Model B (.pth, optional)", required=False),
            _lora(name="lora_b", label="LoRA B"),
            _policy(name="policy_b", label="Guidance B"),
            ArgSpec("label_b", "str", default="B"),
            _scenes(label="Scenes (space-separated NPZ paths)", multi=True, shared=None),
            _output_dir(),
            ArgSpec("steps", "int", default=80),
            ArgSpec("view_half_m", "float", label="View half (m)", default=30.0),
            ArgSpec("ego_wheelbase", "float", label="Ego wheelbase (m)", default=4.76),
            ArgSpec("make_webm", "bool", label="Make WebM", default=True),
            ArgSpec("webm_fps", "int", label="WebM fps", default=10),
            ArgSpec("hist_steps", "int", label="History steps", default=0),
        ],
        outputs=lambda v: {"dir": v.get("output_dir")},
    )
)

# --- Viz: static NPZ-dir render -------------------------------------------------------
_register(
    Workflow(
        key="render_npz_dir",
        title="Viz: render NPZ dir",
        module="scenario_generation.render_npz_dir",
        description="Render a route dir, a single-scene dir, or a parent of collision-window "
        "subfolders to PNGs (perfect-tracker renderer with neighbors). Ego dims are read from "
        "each NPZ — no need to enter them.",
        args=[
            ArgSpec(
                "npz_dir", "dir", label="NPZ dir / route", shared="route_datasets", required=True
            ),
            _output_dir(),
            ArgSpec("route_pkl", "file", label="Route pickle (optional, adds borders/route)"),
            ArgSpec("workers", "int", default=8),
            ArgSpec("stride", "int", default=1),
            ArgSpec("limit", "int", default=-1),
        ],
        outputs=lambda v: {"dir": v.get("output_dir")},
    )
)

# --- Scene Branch Editor (interactive server) -----------------------------------------
_register(
    Workflow(
        key="scene_branch_editor",
        title="Scene Branch Editor",
        module="scenario_generation.tools.scene_branch_editor",
        server=True,
        env="lanelet",  # needs lanelet2 shared libs on LD_LIBRARY_PATH (set before exec)
        description="Interactive obstacle placement + curated-RSFT data authoring. Runs as a "
        "subprocess with the lanelet env and is embedded via iframe.",
        args=[
            ArgSpec(
                "npz_dir",
                "dir",
                label="Replay NPZ dir (route)",
                shared="route_datasets",
                required=True,
            ),
            _model_path(required=False, label="Model (.pth, optional)"),
            ArgSpec(
                "reward_config",
                "file",
                label="Reward config JSON (optional)",
                shared="reward_configs",
            ),
            _ego_shape(required=False),
            ArgSpec("map_path", "file", label="Lanelet2 .osm map (optional)", shared="maps"),
            ArgSpec("tree_json", "file", label="Existing scene tree JSON (optional)"),
            ArgSpec("port", "int", default=7870, label="Port"),
            # Pre-filled by the panel from the workspace (Export = contiguous → routes;
            # RSFT save = individual scenes → scenes). Hidden; injected in _open_editor.
            ArgSpec("export_dir", "dir", default="", hidden=True),
            ArgSpec("rsft_dir", "dir", default="", hidden=True),
        ],
    )
)


def list_workflows() -> list[Workflow]:
    return list(WORKFLOWS.values())


def get_workflow(key: str) -> Workflow:
    return WORKFLOWS[key]
