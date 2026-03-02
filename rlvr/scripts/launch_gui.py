#!/usr/bin/env python3
"""
TeraSim Ghost Replay Launcher GUI.

Simple Gradio interface for browsing a JSON list of NPZ samples and
launching ghost replay simulations in TeraSim.  Designed to be extended
later with GRPO / model-inference panels.

Usage:
    source .venv/bin/activate
    python3 rlvr/scripts/launch_gui.py \\
        --npz_list /media/danielsanchez/.../dpo-npz.json

Opens at http://localhost:7861
"""

import argparse
import json
import math
import time
from pathlib import Path

import gradio as gr

# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parents[2]
_SIM_CONFIG_DIR = _REPO_ROOT / "rlvr" / "sim_config"
_DEFAULT_FCD_DIR = str(Path.home() / "terasim_fcd")

# ---------------------------------------------------------------------------
# Lazy imports for the rlvr package (avoids slow TeraSim startup at module load)
# ---------------------------------------------------------------------------
def _get_bridge_and_utils():
    from rlvr.npz_utils import extract_spawn_states
    from rlvr.terasim_bridge import TeraSimBridge
    return extract_spawn_states, TeraSimBridge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_npz_list(json_path: str) -> list[str]:
    with open(json_path) as f:
        paths = json.load(f)
    if not isinstance(paths, list):
        raise ValueError(f"Expected a JSON list, got {type(paths)}")
    return paths


def _sample_info(npz_path: str) -> str:
    """Return a plain-text summary of the NPZ sample for the info textbox."""
    try:
        extract_spawn_states, _ = _get_bridge_and_utils()
        json_path = npz_path.replace(".npz", ".json")
        spawn = extract_spawn_states(npz_path, json_path)
        ego = spawn["ego"]
        return (
            f"x={ego['x']:.2f}  y={ego['y']:.2f}\n"
            f"yaw={math.degrees(ego['yaw_rad']):.1f}°  "
            f"speed={ego['vx']:.2f} m/s\n"
            f"Active NPCs: {len(spawn['npcs'])}\n"
            f"GT steps: {spawn['ego_future_map'].shape[0]}"
        )
    except Exception as e:
        return f"Could not load: {e}"


def _run_simulation(
    npz_path: str,
    use_gui: bool,
    use_viz: bool,
    use_fcd: bool,
    fcd_dir: str,
    step_delay: float,
):
    """
    Generator: yields log lines while running the ghost replay.
    Called by Gradio on button click; output streams to the log textbox.
    """
    extract_spawn_states, TeraSimBridge = _get_bridge_and_utils()

    json_path = npz_path.replace(".npz", ".json")

    log = ""

    def emit(line: str):
        nonlocal log
        log += line + "\n"
        return log

    yield emit(f"NPZ:  {npz_path}")
    yield emit(f"JSON: {json_path}")
    yield emit("")

    # --- Extract spawn states ---
    yield emit("Extracting spawn states…")
    try:
        spawn = extract_spawn_states(npz_path, json_path)
    except Exception as e:
        yield emit(f"ERROR: {e}")
        return

    ego = spawn["ego"]
    yield emit(
        f"  Ego t=0:  x={ego['x']:.2f}  y={ego['y']:.2f}  "
        f"yaw={math.degrees(ego['yaw_rad']):.1f}°  "
        f"speed={ego['vx']:.2f} m/s"
    )
    yield emit(f"  Active NPCs: {len(spawn['npcs'])}")

    if use_gui:
        yield emit("GUI mode: sumo-gui will open on your desktop.")
        yield emit("  (If no window appears, run:  xhost +local:docker)")
    if use_viz:
        yield emit("Dash visualizer: http://localhost:8050")
    if use_fcd:
        if fcd_dir.startswith("/tmp"):
            yield emit(
                "WARNING: fcd_dir starts with /tmp — Docker bind-mount may not "
                "work on this host. Use a path under /home/ instead."
            )
        yield emit(f"FCD output dir: {fcd_dir}")

    yield emit("")
    yield emit("Starting TeraSim simulation…")

    # --- Run ---
    fcd_host_dir = fcd_dir if use_fcd else None
    try:
        with TeraSimBridge(
            sim_config_host_dir=str(_SIM_CONFIG_DIR),
            gui=use_gui,
            fcd_host_dir=fcd_host_dir,
        ) as sim:
            sim.start_episode(spawn, enable_viz=use_viz)
            yield emit("  Episode started.")

            if use_viz:
                yield emit("  >>> Open http://localhost:8050 in your browser <<<")
                time.sleep(3)

            n_steps = len(spawn["ego_future_map"])
            for step_idx in range(n_steps):
                x, y, yaw_rad = spawn["ego_future_map"][step_idx]
                result = sim.step((float(x), float(y)), float(yaw_rad))

                if step_delay > 0:
                    time.sleep(step_delay)

                if not result["av_in_sim"]:
                    yield emit(
                        f"\nFAILED: AV removed from simulation at step {step_idx} "
                        f"(t={result['sim_time']:.1f}s) — collision or out-of-bounds."
                    )
                    return

                if (step_idx + 1) % 10 == 0:
                    yield emit(
                        f"  step {step_idx + 1:3d}/{n_steps}  "
                        f"t={result['sim_time']:.1f}s  "
                        f"NPCs={len(result['npc_states'])}"
                    )

            # Final position check
            final_state = sim._last_state
            av_state = final_state["agent_details"]["vehicle"]["AV"]
            av_x, av_y = av_state["x"], av_state["y"]
            gt_x, gt_y = float(spawn["ego_future_map"][-1, 0]), float(
                spawn["ego_future_map"][-1, 1]
            )
            dist = math.sqrt((av_x - gt_x) ** 2 + (av_y - gt_y) ** 2)
            yield emit(
                f"\nFinal position:  sim=({av_x:.2f}, {av_y:.2f})  "
                f"GT=({gt_x:.2f}, {gt_y:.2f})  error={dist:.3f}m"
            )

            if dist >= 2.0:
                yield emit(
                    f"\nFAILED: position error {dist:.3f}m > 2.0m threshold."
                )
                return

            # FCD path
            fcd_path = sim.fcd_output_path
            if fcd_path:
                fcd = Path(fcd_path)
                if fcd.exists():
                    yield emit(
                        f"\nFCD written: {fcd_path}  ({fcd.stat().st_size // 1024} KB)"
                    )
                    yield emit(
                        f"Replay:  python3 rlvr/scripts/replay_fcd.py "
                        f"--fcd_file {fcd_path}"
                    )

            yield emit("\n✓  Ghost replay validation PASSED")

    except Exception as e:
        yield emit(f"\nERROR: {e}")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_interface(npz_paths: list[str]) -> gr.Blocks:
    total = len(npz_paths)

    # Shared mutable state
    state = {"index": 0}

    def _clamp(i: int) -> int:
        return max(0, min(total - 1, i))

    def _load_index(i: int):
        """Update displayed info when index changes."""
        i = _clamp(int(i) - 1)      # UI is 1-based
        state["index"] = i
        path = npz_paths[i]
        info = _sample_info(path)
        # returns: displayed_index, current_path, info_text, hidden_path
        return i + 1, path, info, path

    def _nav(delta: int, current_displayed: int):
        new = _clamp(int(current_displayed) - 1 + delta)
        return _load_index(new + 1)

    with gr.Blocks(title="TeraSim Ghost Replay") as demo:
        gr.Markdown("# TeraSim Ghost Replay Launcher")
        gr.Markdown(
            f"Loaded **{total}** samples.  "
            "Browse by index, configure options, then launch the simulation."
        )

        # Hidden component to pass the resolved NPZ path to the simulation
        npz_path_state = gr.Textbox(visible=False)

        with gr.Row():
            # ── LEFT: sample browser ────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### NPZ Browser")

                with gr.Row():
                    index_input = gr.Number(
                        label="Sample index",
                        value=1,
                        minimum=1,
                        maximum=total,
                        precision=0,
                    )
                    gr.Markdown(f"/ {total}")

                with gr.Row():
                    btn_prev = gr.Button("◄ Prev", size="sm")
                    btn_next = gr.Button("Next ►", size="sm")

                # Visible textbox so the user can clearly see the current path
                current_path_box = gr.Textbox(
                    label="Current NPZ path",
                    value=npz_paths[0],
                    interactive=False,
                    lines=2,
                )

                sample_info_md = gr.Textbox(
                    label="Sample info",
                    value=_sample_info(npz_paths[0]),
                    interactive=False,
                    lines=6,
                )

            # ── RIGHT: simulation options ────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### Simulation Options")

                with gr.Row():
                    use_gui = gr.Checkbox(
                        label="sumo-gui  (--gui)",
                        value=False,
                        info="Opens sumo-gui on your desktop via X11. "
                             "Run `xhost +local:docker` first.",
                    )
                    use_viz = gr.Checkbox(
                        label="Dash web viewer  (--viz)",
                        value=False,
                        info="When checked, open http://localhost:8050 in a "
                             "separate browser tab after clicking Launch.",
                    )

                gr.Markdown(
                    "💡 **Dash simulation view → [http://localhost:8050](http://localhost:8050)**  "
                    "(only active while a simulation is running with --viz checked)"
                )

                with gr.Row():
                    use_fcd = gr.Checkbox(
                        label="Record FCD output",
                        value=False,
                        info="Write SUMO FCD trajectory file to disk.",
                    )
                    fcd_dir = gr.Textbox(
                        label="FCD output directory",
                        value=_DEFAULT_FCD_DIR,
                        placeholder="/home/user/terasim_fcd",
                        interactive=True,
                    )

                step_delay = gr.Slider(
                    label="Step delay (s)  — 0 = as fast as possible, 0.1 = real-time",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.1,
                )

        # ── Launch button ────────────────────────────────────────────────
        launch_btn = gr.Button(
            "🚀  Launch TeraSim Simulation",
            variant="primary",
            size="lg",
        )

        # ── Output log ───────────────────────────────────────────────────
        gr.Markdown("### Output")
        output_log = gr.Textbox(
            label="Simulation log",
            lines=20,
            max_lines=40,
            interactive=False,
        )

        # ── Event wiring ─────────────────────────────────────────────────
        # Order matches _load_index return: (index, path_box, info, hidden_path)
        _outputs = [index_input, current_path_box, sample_info_md, npz_path_state]

        index_input.submit(fn=_load_index, inputs=[index_input], outputs=_outputs)
        btn_prev.click(fn=lambda i: _nav(-1, i), inputs=[index_input], outputs=_outputs)
        btn_next.click(fn=lambda i: _nav(+1, i), inputs=[index_input], outputs=_outputs)

        # Initialise state on load
        demo.load(fn=lambda: _load_index(1), outputs=_outputs)

        launch_btn.click(
            fn=_run_simulation,
            inputs=[npz_path_state, use_gui, use_viz, use_fcd, fcd_dir, step_delay],
            outputs=[output_log],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TeraSim Ghost Replay Launcher GUI."
    )
    parser.add_argument(
        "--npz_list",
        required=True,
        help="JSON file containing a list of .npz paths.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7861,
        help="Gradio server port (default 7861; 7860 is used by the DPO GUI).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link.",
    )
    args = parser.parse_args()

    print(f"Loading NPZ list from {args.npz_list}…")
    npz_paths = _load_npz_list(args.npz_list)
    print(f"  {len(npz_paths)} samples loaded.")

    demo = build_interface(npz_paths)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
