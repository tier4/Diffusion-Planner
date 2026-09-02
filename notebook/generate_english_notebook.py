"""Generate the English token-analysis notebook from the Japanese source.

Code cells are copied verbatim so both notebooks always perform the same
analysis. Only explanatory Markdown and user-facing code-cell text are
translated.
"""

import copy
import json
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "token_analysis.ipynb"
OUTPUT = HERE / "token_analysis_en.ipynb"


def md(text):
    return textwrap.dedent(text).strip()


MARKDOWN = {
    "7fb27b941602401d91542211134fc71a": md(
        """
        # Token Importance / Attention Analysis

        ## Purpose

        Diffusion Planner receives not only ego history but also many tokens representing
        neighboring agents, lanes, routes, road boundaries, and other context. This experiment
        determines **which inputs the trained model actually uses**.

        It addresses four questions:

        1. How much does prediction accuracy degrade when neighbors or lanes are removed?
        2. Are all 320 neighbor and 140 lane slots necessary, or are nearby tokens sufficient?
        3. Which token classes receive attention inside the Fusion Encoder?
        4. Are distant tokens useful, or do they behave like noise?

        These results help identify candidates for reducing token capacity and inference cost.
        They describe the behavior of **this checkpoint on the selected evaluation data**; they
        do not prove that an input can be removed safely. Final decisions require retraining and
        closed-loop evaluation.

        ## Running the analysis

        `run_token_analysis.sh` runs the analyses and builds both Japanese and English reports.

        ```bash
        MODEL_DIR=best_models/20260730/best_model \\
        DATADIR=/path/to/dataset \\
        N_SAMPLES=1024 BATCH_SIZE=64 DEVICE=cuda \\
        ./run_token_analysis.sh
        ```

        The workflow:

        1. Run input ablations with `scripts/token_importance.py` and save FDE/ADE to TSV.
        2. Aggregate attention and valid-token statistics with `scripts/attention_analysis.py`.
        3. Execute this notebook and create a self-contained HTML report.

        Plot images and styles are embedded in the HTML, so the report can be viewed on another
        machine without the repository, TSV, or JSON files.
        """
    ),
    "experiment-guide-01": md(
        """
        ## Feature Importance Method (Input Ablation)

        This is not tree-model feature importance. It is an **ablation test that measures how
        prediction error changes when selected input information is hidden**.

        ### Procedure

        1. Select evaluation scenes whose displacement is at least `MOVE_MIN_M`.
        2. Run the unmodified input and compute baseline FDE/ADE.
        3. Run the same scenes again after replacing only the target input with zeros.
        4. Record `ablation error - baseline error` as importance (delta).

        The same scenes, model, and initial trajectory are used, making paired differences easy
        to interpret.

        ### Class-level drop

        `drop:neighbors` and `drop:lanes` zero an entire input class.

        - Variable-length inputs such as neighbors and maps become padding and are excluded by
          the attention mask.
        - Goal pose and turn indicators are fixed tokens. Their information is replaced by a
          constant, but the token itself remains.

        ### Nearest Top-K

        `nbr_top:16` keeps only the 16 neighbors closest to the ego. The smallest K whose error
        converges to baseline is a candidate token capacity. Distance is current Euclidean
        distance; it does not rank by future collision risk or road connectivity.

        ### Radius cutoff

        `nbr_within:50` keeps neighbors within 50 m. Top-K studies count limits, whereas radius
        cutoffs study the physical input range.

        ### Interpreting the delta

        - **Delta > 0:** removing the input hurts accuracy; the checkpoint likely uses it.
        - **Delta near 0:** no clear contribution is visible for this data and metric.
        - **Delta < 0:** removal improves accuracy; possible causes include noisy/misused input,
          sampling variation, or distribution shift.

        A near-zero delta does not prove that an input is unnecessary. Effects on rare
        safety-critical scenes may disappear in mean FDE/ADE, and zero-filled input can be
        out-of-distribution. Confirm any reduction with more samples, closed-loop safety metrics,
        and retraining with the reduced input.
        """
    ),
    "parameter-guide-01": md(
        """
        ## Metrics and Runtime Parameters

        ### FDE / ADE

        - **FDE (Final Displacement Error):** distance between predicted and ground-truth final
          positions. Lower is better.
        - **ADE (Average Displacement Error):** mean position error over all predicted timesteps.
          Lower is better.
        - **Delta FDE / ADE:** ablation result minus baseline. Positive means degradation.

        ### Main environment variables

        | Variable | Default | Meaning |
        |---|---:|---|
        | `MODEL_DIR` | `best_models/20260730/best_model` | Directory containing `args.json` and `best_model.pth` |
        | `DATADIR` | mini dataset | Evaluation dataset root |
        | `VALID_LIST` | `$DATADIR/path_list_valid.json` | List of evaluation NPZ files |
        | `N_SAMPLES` | 128 | Maximum number of moving scenes |
        | `BATCH_SIZE` | 32 | Inference batch size |
        | `DEVICE` | `cuda` | `cuda` or `cpu` |
        | `MOVE_MIN_M` | 5.0 | Minimum endpoint displacement for scene selection |
        | `TURN_DEG` | 15.0 | Endpoint bearing threshold for the turning subset |
        | `OUT_DIR` | generated from dataset name | Output directory |

        Start with `N_SAMPLES=128` for a smoke test, then use 512-1024 or more for formal
        evaluation. Use the same data list and thresholds when comparing runs.
        """
    ),
    "importance-full-table-guide": md(
        """
        ### Complete Feature-Importance Results

        This table shows every configuration and every metric stored in the TSV. `top` evaluates
        the trajectory selected by the model; `min` evaluates the candidate closest to ground
        truth. They are identical for single-trajectory output. Each delta is relative to the
        baseline value of the same metric.
        """
    ),
    "8dd0d8092fe74a7c96281538738b07e2": md(
        """
        ## 1. Feature Importance by Input Class

        The two panels show delta FDE and delta ADE after hiding an entire input class. FDE
        measures the final point, whereas ADE measures the whole path.

        - A longer positive bar means stronger degradation and likely stronger dependency.
        - A value near zero means no clear effect was observed for this data and metric.
        - A negative value means removal improved the metric, but is not sufficient evidence to
          delete the input.
        - A large delta FDE only suggests endpoint impact; a large delta ADE only suggests impact
          on the intermediate path.

        Compare both sign and magnitude, and check whether small differences reproduce across
        sample sizes and splits. Goal pose and turn indicators are constant-value ablations, not
        complete token removal.
        """
    ),
    "8edb47106e1a46a883d545849b8ab81b": md(
        """
        ## 2. Nearest Top-K and Radius Cutoffs

        K is the number of retained tokens. The top row shows FDE, the bottom row shows ADE, and
        dashed lines show the all-token baseline.

        ### Reading the curves

        - Error falls as K grows: additional tokens provide useful information.
        - Both FDE and ADE stabilize near baseline: that K is a capacity candidate.
        - Error grows as K grows: distant tokens may add noise.
        - FDE and ADE converge at different K values: endpoint and path quality may require
          different capacities.
        - Irregular curves may indicate insufficient samples, unsuitable distance ranking, or
          out-of-distribution ablations.

        Define acceptable delta FDE and delta ADE before choosing the smallest K. Top-K and radius
        cutoffs answer different design questions and need not give the same conclusion.
        """
    ),
    "8763a12b2bbd4a93a75aff182afb95dc": md(
        """
        ## 3. Attention Analysis and Interpretation

        ### What attention represents

        In the Fusion Encoder, each token (query) assigns weights to other tokens (keys). Weights
        over valid keys sum to one for each query. This report averages weights across heads.

        Attention shows where the model reads information; it does not directly measure how many
        meters that information changes the final prediction. Interpret it together with
        ablation importance.

        ### Reported quantities

        - **count share:** fraction of all valid tokens belonging to the class
        - **ego-query share:** attention sent from the ego token to the class
        - **all-query share:** mean attention received from all valid queries
        - **selectivity:** `attention share / count share`
        - **value-weighted share:** attention multiplied by value-vector magnitude; an
          approximation that omits per-head structure and output projection

        ### Selectivity

        - Near **1.0:** attention is roughly proportional to token count.
        - Above **1.0:** the class is preferred beyond its count share.
        - Below **1.0:** the class receives less attention than its count share.

        Always inspect both selectivity and absolute share. A rare class can have high
        selectivity but little total attention; a numerous class can dominate total attention
        despite selectivity below one.

        ### Layers, distance, and turning subsets

        - A class whose share increases in deeper layers may be integrated later.
        - Distance bins show the within-class attention distribution and include only scenes
          where that class is present.
        - Distance-bin values are not normalized by token count per bin.
        - Turning/straight route share is descriptive; the smaller subset can be unstable.

        ### Combining attention and ablation

        | Attention | Ablation degradation | Candidate interpretation |
        |---|---|---|
        | High | Large | Strongly read and influential |
        | High | Small | Read but redundant, or weak effect on output |
        | Low | Large | Sparse but important information, or effect through another layer/query |
        | Low | Small | Limited use under this checkpoint and evaluation |

        These are diagnostic hypotheses, not causal proof. Combine reproducibility, Top-K,
        radius cutoffs, and closed-loop safety evaluation before reducing inputs.
        """
    ),
    "token-occupancy-guide": md(
        """
        ## 4. Valid Token Counts and Slot Utilization

        Valid-token counts are computed from the padding mask during attention analysis and saved
        under `token_occupancy` in the attention JSON. They use the same selected moving scenes;
        no separate dataset scan is required.

        - **slots:** maximum allocated model slots
        - **mean:** mean valid count per scene
        - **p50:** half of scenes have this count or fewer
        - **p95 / p99:** 95% / 99% of scenes have this count or fewer
        - **max:** largest count observed in this run
        - **mean utilization:** `mean / slots * 100`

        p95 or p99 can seed a capacity proposal. `p99_capacity` is the smallest integer that
        covers 99% of this sample, but dropping the remaining 1% is not automatically safe.
        Combine occupancy with Top-K behavior, the truncation rule, and closed-loop evaluation.

        Utilization measures occupancy, not usefulness. Rare tokens may be important, while a
        densely occupied class may still be redundant.
        """
    ),
    "938c804e27f84196a10c8828c723f798": md(
        """
        ## 5. Interpretation Checklist

        1. Are baseline FDE/ADE plausible, and is the sample count sufficient?
        2. Are class-drop deltas practically meaningful, not merely nonzero?
        3. Do trends reproduce across sample sizes and data splits?
        4. Where do Top-K curves stabilize, and do radius-cutoff results agree?
        5. Do mean, p95, p99, and maximum counts expose rare dense scenes?
        6. Is attention share explained only by token count? Check selectivity.
        7. Why do ego-query, all-query, and value-weighted shares differ?
        8. Do attention and ablation agree? Avoid forcing a useful/useless conclusion when they do
           not.
        9. Have rare safety-critical scenes been checked individually and in closed loop?

        This notebook supports conclusions about **input dependency, attention patterns, and token
        occupancy for the current checkpoint on the selected data**. Claims about general input
        necessity or safety after removal require retraining, multiple splits, and closed-loop
        evaluation.
        """
    ),
}


CODE_REPLACEMENTS = {
    "# Shell実行時は環境変数から成果物を受け取る。対話実行時は既定値を使用。": (
        "# Read result paths from the shell; use defaults for interactive execution."
    ),
    "NOT FOUND (パスを設定してください)": "NOT FOUND (set the result path)",
    "NOT FOUND (任意)": "NOT FOUND (optional)",
    "# 検証済みパレット(dataviz reference)。系列色はエンティティ固定": (
        "# Fixed colors for token classes across all plots."
    ),
    "# 劣化(+Δ)": "# degradation (+delta)",
    "# 改善(−Δ)": "# improvement (-delta)",
    "注: この結果ではtopとminが全設定で同値（単一trajectory出力）です。": (
        "Note: top and min are identical for every configuration (single-trajectory output)."
    ),
    "距離カットオフ:": "Radius cutoffs:",
}


def main():
    notebook = json.loads(SOURCE.read_text())
    notebook = copy.deepcopy(notebook)

    for cell in notebook["cells"]:
        cell_id = cell.get("id")
        if cell["cell_type"] == "markdown":
            if cell_id not in MARKDOWN:
                raise KeyError(f"missing English Markdown for cell {cell_id}")
            cell["source"] = MARKDOWN[cell_id]
        elif cell["cell_type"] == "code":
            source = cell["source"]
            as_list = isinstance(source, list)
            text = "".join(source) if as_list else source
            for old, new in CODE_REPLACEMENTS.items():
                text = text.replace(old, new)
            cell["source"] = text.splitlines(keepends=True) if as_list else text

        cell["execution_count"] = (
            None if cell["cell_type"] == "code" else cell.get("execution_count")
        )
        if cell["cell_type"] == "code":
            cell["outputs"] = []

    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
