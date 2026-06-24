"""Unified Gradio control panel: one app, a central asset library, one tab per workflow.

Assets (models, guidance policies, datasets, reward configs, maps) are registered ONCE in the
Workspace tab; every workflow form picks them from dropdowns. The panel adds no domain logic —
Run shells out to the existing CLI tools and streams their output. The Scene Editor runs
in-process (a Gradio sub-server in a daemon thread) and is shown in its tab via an iframe.

Launch:  python -m control_panel   (--port / --host / --editor_port / --share)
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import gradio as gr

from . import presets as P
from . import runner as R
from . import workflows as W

CUSTOM = "custom…"
NONE = "(none)"

# Holds the in-process Scene Editor sub-server so Reload can close the previous one.
_EDITOR: dict = {"demo": None, "port": None}


# --------------------------------------------------------------------------------------
# Value coercion + form building
# --------------------------------------------------------------------------------------
def _coerce(spec: W.ArgSpec, v):
    if spec.kind == "bool":
        return bool(v)
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    if spec.kind == "int":
        return int(round(float(v)))
    if spec.kind == "float":
        return float(v)
    return v


def _field_role(spec: W.ArgSpec) -> str:
    if spec.hidden:
        return "hidden"
    if spec.derive_from:
        return "derive"
    if spec.shared == "ego_shape":
        return "ego"
    if spec.shared == "output_dir":
        return "outdir"
    if spec.shared in P.LIST_TYPES:
        return "list"
    return "plain"


def _list_choices(library: dict, asset_type: str, required: bool) -> list[str]:
    base = P.entry_names(library, asset_type)
    return ([] if required else [NONE]) + base + [CUSTOM]


def _plain_widget(spec: W.ArgSpec, default=None):
    label = spec.label + (" *" if spec.required else "")
    info = spec.help or None
    d = spec.default if default is None else default
    if spec.kind == "bool":
        return gr.Checkbox(value=bool(spec.default), label=label, info=info)
    if spec.kind == "choice":
        return gr.Dropdown(
            choices=spec.choices or [],
            value=d or (spec.choices or [None])[0],
            label=label,
            info=info,
        )
    if spec.kind in ("int", "float"):
        return gr.Number(
            value=d if d not in ("", None) else None,
            label=label,
            info=info,
            precision=0 if spec.kind == "int" else None,
        )
    return gr.Textbox(value="" if d in (None,) else str(d), label=label, info=info)


def build_form(wf: W.Workflow, library: dict, asset_dropdowns: dict):
    """Render the form. Returns (fields, flat_comps).

    fields: list of {name, spec, role, comps}. Shared-list fields render a dropdown over the
    library + a custom-path textbox; derive/ego fields render nothing (resolved at Run).
    asset_dropdowns[type] collects dropdowns for cross-tab refresh.
    """
    fields, flat = [], []
    for spec in wf.args:
        role = _field_role(spec)
        comps = []
        if role == "hidden":
            comps = []  # not rendered; resolve uses spec.default
        elif role == "list":
            names = _list_choices(library, spec.shared, spec.required)
            base = P.entry_names(library, spec.shared)
            value = (
                names[0] if (spec.required and base) else (NONE if not spec.required else CUSTOM)
            )
            dd = gr.Dropdown(
                choices=names,
                value=value,
                label=spec.label + (" *" if spec.required else "") + f"  ({spec.shared})",
                info=spec.help or None,
            )
            custom = gr.Textbox(
                value="", label=f"{spec.label} — custom path", visible=(value == CUSTOM)
            )
            dd.change(lambda v: gr.update(visible=(v == CUSTOM)), dd, custom)
            asset_dropdowns.setdefault(spec.shared, []).append((dd, spec.required))
            comps = [dd, custom]
        elif role == "outdir":
            comps = [
                gr.Textbox(
                    value=library.get("output_dir", ""),
                    label=spec.label + (" *" if spec.required else ""),
                    info=spec.help or None,
                )
            ]
        elif role in ("derive", "ego"):
            comps = []  # not rendered; resolved from library / sibling model at Run
        else:  # plain
            comps = [_plain_widget(spec)]
        fields.append({"name": spec.name, "spec": spec, "role": role, "comps": comps})
        flat.extend(comps)
    return fields, flat


def resolve_values(wf: W.Workflow, fields: list, library: dict, flat_values: tuple) -> dict:
    """Reconstruct the CLI values dict from the flat form values + the live library."""
    it = iter(flat_values)
    values: dict = {}
    model_entries: dict = {}  # arg name -> selected model library entry
    for f in fields:
        spec, role, name = f["spec"], f["role"], f["name"]
        vals = [next(it) for _ in f["comps"]]
        if role == "hidden":
            values[name] = spec.default
        elif role == "ego":
            values[name] = library.get("ego_shape", "")
        elif role == "outdir":
            values[name] = vals[0] if (vals[0] not in (None, "")) else library.get("output_dir", "")
        elif role == "list":
            sel, custom = vals[0], vals[1]
            if sel in (NONE, None, ""):
                values[name] = ""
                entry = {}
            elif sel == CUSTOM:
                values[name] = (custom or "").strip()
                entry = {}
            else:
                entry = P.find_entry(library, spec.shared, sel) or {}
                values[name] = entry.get("path", "")
            if spec.shared == "models":
                model_entries[name] = entry
        elif role == "derive":
            pass  # second pass
        else:
            values[name] = _coerce(spec, vals[0])
    for f in fields:
        if f["role"] != "derive":
            continue
        spec = f["spec"]
        entry = model_entries.get(spec.derive_from, {})
        values[f["name"]] = entry.get(spec.derive_field, "") or ""
    return values


# --------------------------------------------------------------------------------------
# Run / preview / stop / attach handlers
# --------------------------------------------------------------------------------------
def _err(msg: str) -> str:
    return f"<span style='color:#d33'>⚠ {msg}</span>"


def _run_handler(wf, fields):
    def run(library, *flat):
        values = resolve_values(wf, fields, library, flat)
        try:
            job = R.launch(wf, values)
        except ValueError as e:  # missing required asset(s)
            yield "", "not started", _err(str(e)), None
            return
        yield f"Launched PID {job.pid}\n  log: {job.logfile}\n", f"running (PID {job.pid})", "", job
        for text in R.stream(job):
            yield text, f"running (PID {job.pid})", "", job
        yield R.read_log(job), f"finished (alive={R.is_alive(job.pid)})", "", job

    return run


def _preview_handler(wf, fields):
    def preview(library, *flat):
        values = resolve_values(wf, fields, library, flat)
        try:
            return "$ " + shlex.join(R.build_full_command(wf, values))
        except ValueError as e:
            return _err(str(e))

    return preview


def _stop_handler():
    def stop(job):
        if not job:
            return "no active job to stop"
        ok = R.stop(job)
        return f"stopped PID {job.pid}" if ok else f"stop failed for PID {job.pid}"

    return stop


def _attach_handler(key):
    def attach():
        job = R.latest_job(key)
        if job is None:
            yield "No prior job for this workflow.", "none", None
            return
        for text in R.stream(job):
            yield text, f"attached PID {job.pid} (alive={R.is_alive(job.pid)})", job
        yield R.read_log(job), f"finished PID {job.pid}", job

    return attach


def workflow_panel(wf: W.Workflow, library0: dict, library_state, asset_dropdowns: dict):
    """Standard form + buttons + live log for one workflow. Returns key refs."""
    gr.Markdown(f"**{wf.title}** — {wf.description}")
    fields, flat = build_form(wf, library0, asset_dropdowns)
    job_state = gr.State(None)
    with gr.Row():
        run_btn = gr.Button("▶ Run", variant="primary")
        stop_btn = gr.Button("■ Stop")
        prev_btn = gr.Button("Preview command")
        attach_btn = gr.Button("Attach to latest")
    err = gr.Markdown()
    status = gr.Textbox(label="Status", interactive=False)
    log = gr.Textbox(label="Log", lines=20, max_lines=20, autoscroll=True, interactive=False)

    run_btn.click(_run_handler(wf, fields), [library_state, *flat], [log, status, err, job_state])
    stop_btn.click(_stop_handler(), job_state, status)
    prev_btn.click(_preview_handler(wf, fields), [library_state, *flat], log)
    attach_btn.click(_attach_handler(wf.key), None, [log, status, job_state])
    return {"fields": fields, "flat": flat, "log": log, "status": status}


# --------------------------------------------------------------------------------------
# Tab extras
# --------------------------------------------------------------------------------------
_EPOCH_HEADERS = ["epoch", "split", "n", "reward", "rb_cross", "lane_dep", "sc_mean", "cl_mean"]


def _epoch_table(key: str):
    job = R.latest_job(key)
    if job is None:
        return [], "No training run found yet."
    rows = R.parse_epoch_metrics(job.logfile)
    if not rows:
        return [], f"No epoch metrics parsed yet (job {job.started_at})."
    return [
        [r.get(h, "") for h in _EPOCH_HEADERS] for r in rows
    ], f"job {job.started_at}: {len(rows)} rows"


def _scan_outputs(output_dir: str):
    if not output_dir or not Path(output_dir).exists():
        return None, [], "Output dir not found."
    root = Path(output_dir)
    webms = sorted(root.rglob("*.webm"))
    pngs = sorted(root.rglob("*.png"))
    video = str(webms[0]) if webms else None
    return video, [str(p) for p in pngs[:60]], f"{len(webms)} webm, {len(pngs)} png."


def _baseline_md(library: dict) -> str:
    bm = library.get("baseline_metrics", {}) or {}
    if not any(v is not None for v in bm.values()):
        return "_Baseline column unset — fill `baseline_metrics` in the library file._"
    rows = "\n".join(f"| {k} | {v} |" for k, v in bm.items())
    return "**Frozen baseline (never recomputed):**\n\n| metric | value |\n|---|---|\n" + rows


def _resolve_asset(asset_type: str, sel_path: str) -> tuple[dict, str]:
    """Turn a browsed path into a smart library entry.

    Models: if a dir, pick best_model.pth (else first *.pth); auto-attach a sibling args.json
    (same dir or parent) and a lora_dir if an adapter is alongside. Maps: find the .osm if a
    dir was picked. Returns (entry-without-name, note describing what was auto-detected).
    """
    p = Path(sel_path)
    entry: dict = {"path": sel_path}
    notes: list[str] = []
    if asset_type == "models":
        pth = p
        if p.is_dir():
            cands = list(p.glob("best_model.pth")) or sorted(p.glob("*.pth"))
            if cands:
                pth = cands[0]
                entry["path"] = str(pth)
                notes.append(f"model={pth.name}")
        for cand in (pth.parent / "args.json", pth.parent.parent / "args.json"):
            if cand.exists():
                entry["args_json"] = str(cand)
                notes.append("args.json ✓")
                break
        else:
            notes.append("⚠ no args.json found nearby")
        if (pth.parent / "adapter_config.json").exists():
            entry["lora_dir"] = str(pth.parent)
            notes.append("lora ✓")
    elif asset_type == "maps" and p.is_dir():
        osm = sorted(p.glob("*.osm"))
        if osm:
            entry["path"] = str(osm[0])
            notes.append(f"map={osm[0].name}")
    return entry, (", ".join(notes) if notes else "")


_TYPE_COLOR = {
    "models": "#2563eb",
    "loras": "#7c3aed",
    "policies": "#0891b2",
    "datasets": "#16a34a",
    "reward_configs": "#d97706",
    "maps": "#dc2626",
    "run_dirs": "#475569",
}


def _library_html(library: dict) -> str:
    """Styled card view of the asset library."""
    import html as _h

    out = ["<div style='font-family:system-ui,sans-serif;font-size:13px'>"]
    total = 0
    for t in P.LIST_TYPES:
        entries = library.get(t, [])
        if not entries:
            continue
        total += len(entries)
        c = _TYPE_COLOR.get(t, "#475569")
        out.append(
            f"<div style='margin:10px 0 4px'><span style='background:{c};color:#fff;"
            f"padding:2px 10px;border-radius:12px;font-weight:600'>{t} · {len(entries)}</span></div>"
        )
        for e in entries:
            badges = "".join(
                f"<span style='background:#eef;color:#334;border-radius:6px;padding:1px 6px;"
                f"margin-left:6px;font-size:11px'>{k}</span>"
                for k in ("args_json", "lora_dir", "role")
                if e.get(k)
            )
            path = _h.escape(e.get("path", ""))
            name = _h.escape(e.get("name", "?"))
            out.append(
                f"<div style='border-left:3px solid {c};padding:3px 10px;margin:3px 0 3px 4px'>"
                f"<b>{name}</b>{badges}<br>"
                f"<code style='font-size:11px;color:#777'>{path}</code></div>"
            )
    es = _h.escape(library.get("ego_shape", ""))
    od = _h.escape(library.get("output_dir", ""))
    out.append(
        f"<div style='margin-top:10px;color:#555'>ego_shape <code>{es}</code> · "
        f"output_dir <code>{od}</code></div>"
    )
    if total == 0:
        return "<i>Library empty — browse a file and click an <b>Add</b> button.</i>"
    return "".join(out) + "</div>"


def _list_dir(path: str, query: str = "") -> tuple[list, str]:
    """Navigable directory listing for the file browser.

    Returns (radio choices as (label, full_path), normalised dir). Folders first, then files;
    a leading '⬆ ..' entry; substring filter; capped so huge dirs stay responsive.
    """
    p = Path(path).expanduser() if path else Path.home()
    if not p.is_dir():
        p = p.parent if p.parent.is_dir() else Path.home()
    q = (query or "").lower()
    dirs, files = [], []
    try:
        for e in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            if q and q not in e.name.lower():
                continue
            try:
                is_dir = e.is_dir()
            except OSError:
                continue
            (dirs if is_dir else files).append(
                (f"📁 {e.name}/" if is_dir else f"📄 {e.name}", str(e))
            )
    except (PermissionError, OSError):
        pass
    choices = []
    if p.parent != p:
        choices.append(("⬆ ..", str(p.parent)))
    choices += dirs[:400] + files[:400]
    return choices, str(p)


def _scan_run(run_dir: str):
    """List registerable checkpoints in a training run dir."""
    if not run_dir or not Path(run_dir).exists():
        return gr.update(choices=[], value=None), "Run dir not found.", {}
    root = Path(run_dir)
    found: dict[str, dict] = {}
    latest = root / "latest.pth"
    args_json = root / "args.json"
    base = str(latest) if latest.exists() else ""
    aj = str(args_json) if args_json.exists() else ""
    for d in sorted(root.glob("lora_epoch_*")):
        if d.is_dir():
            found[f"LoRA {d.name}"] = {"path": base, "args_json": aj, "lora_dir": str(d)}
    for f in ("merged.pth", "best_model.pth"):
        if (root / f).exists():
            found[f] = {"path": str(root / f), "args_json": aj, "lora_dir": ""}
    for d in sorted(root.glob("epoch_*")):
        bm = d / "best_model.pth"
        if bm.exists():
            found[f"{d.name}/best_model.pth"] = {"path": str(bm), "args_json": aj, "lora_dir": ""}
    cfg = {}
    gc = root / "grpo_config.json"
    if gc.exists():
        try:
            cfg = json.loads(gc.read_text())
        except (json.JSONDecodeError, OSError):
            cfg = {"error": "could not read grpo_config.json"}
    msg = f"{len(found)} checkpoints found." if found else "No checkpoints found."
    return (
        gr.update(choices=list(found), value=(next(iter(found), None))),
        msg,
        {"found": found, "cfg": cfg},
    )


# --------------------------------------------------------------------------------------
# Scene Editor (in-process sub-server)
# --------------------------------------------------------------------------------------
def _open_editor(library, host, editor_port, fields, *flat):
    values = resolve_values(W.get_workflow("scene_branch_editor"), fields, library, flat)
    npz_dir = values.get("npz_dir")
    if not npz_dir:
        return "", "⚠ Set the Replay NPZ dir before opening the editor."
    try:
        from scenario_generation.tools.scene_branch_editor import build_demo_from_paths
    except Exception as e:  # noqa: BLE001 - surface import/env issues to the UI
        return "", f"⚠ Could not import scene editor: {e}"
    prev = _EDITOR.get("demo")
    if prev is not None:
        try:
            prev.close()
        except Exception:
            pass
    demo = build_demo_from_paths(
        npz_dir=npz_dir,
        model_path=values.get("model_path") or None,
        reward_config=values.get("reward_config") or None,
        ego_shape=values.get("ego_shape") or None,
        map_path=values.get("map_path") or None,
        tree_json=values.get("tree_json") or None,
    )
    port = int(editor_port)
    demo.launch(server_name="0.0.0.0", server_port=port, prevent_thread_lock=True, quiet=True)
    _EDITOR["demo"], _EDITOR["port"] = demo, port
    url = f"http://{host or 'localhost'}:{port}"
    html = (
        f'<p><a href="{url}" target="_blank">Open in new tab ↗</a> (give it a few seconds)</p>'
        f'<iframe src="{url}" width="100%" height="900" '
        'style="border:1px solid #ccc;border-radius:6px"></iframe>'
    )
    return html, f"editor on {url}"


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------
def build_app(host: str = "localhost", default_editor_port: int = 7899) -> gr.Blocks:
    library0 = P.load_library()
    wf = W.get_workflow
    asset_dropdowns: dict[str, list] = {}

    with gr.Blocks(title="Autoresearch Control Panel") as demo:
        gr.Markdown(
            "# Autoresearch Control Panel\n"
            "Register assets once in **Workspace**; pick them from dropdowns in each tab. "
            "Run launches a detached subprocess (survives closing this panel); ■ Stop ends it."
        )
        library_state = gr.State(library0)

        # ---- Workspace -------------------------------------------------------------
        with gr.Tab("Workspace"):
            gr.Markdown(
                "### File browser — navigate your PC / SSDs, select a file or folder, then **Add**"
            )
            _root = library0.get("ssd_root") or str(Path.home())
            _init_choices, _root = _list_dir(_root)
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        fb_path = gr.Textbox(
                            value=_root, label="Folder (type a path + Enter to jump)", scale=4
                        )
                        fb_up = gr.Button("⬆ Up", scale=1)
                    fb_query = gr.Textbox(value="", label="Filter (substring)")
                    fb_list = gr.Radio(
                        choices=_init_choices, label="Contents (📁 folders / 📄 files)"
                    )
                    fb_open = gr.Button("Open selected folder")
                    fb_selected = gr.Textbox(label="Selected path", interactive=False)
                with gr.Column(scale=1):
                    add_name = gr.Textbox(label="Name for the selected asset")
                    gr.Markdown("Adds the **selected path** (or current folder if none selected):")
                    add_model = gr.Button("Add as Model", variant="primary")
                    add_lora = gr.Button("Add as LoRA")
                    add_policy = gr.Button("Add as Guidance policy")
                    add_dataset = gr.Button("Add as Dataset")
                    add_reward = gr.Button("Add as Reward config")
                    add_map = gr.Button("Add as Map")
                    add_run = gr.Button("Add as Run dir")
                    ws_status = gr.Textbox(label="", interactive=False)
            with gr.Row():
                ego_box = gr.Textbox(
                    value=library0.get("ego_shape", ""), label="Ego shape (WB,L,W) — global"
                )
                out_box = gr.Textbox(
                    value=library0.get("output_dir", ""), label="Default output dir"
                )
            remove_dd = gr.Dropdown(label="Remove entry (type/name)", choices=[])
            remove_btn = gr.Button("Remove selected entry")
            gr.Markdown("### Loaded assets")
            lib_view = gr.HTML(value=_library_html(library0))

            # file-browser navigation
            def _refresh_list(path, query):
                choices, norm = _list_dir(path, query)
                return gr.update(choices=choices, value=None), norm

            fb_path.submit(_refresh_list, [fb_path, fb_query], [fb_list, fb_path])
            fb_query.change(_refresh_list, [fb_path, fb_query], [fb_list, fb_path])
            fb_list.change(lambda v: v or "", fb_list, fb_selected)

            def _go_up(path):
                parent = str(Path(path).expanduser().parent)
                choices, norm = _list_dir(parent)
                return gr.update(choices=choices, value=None), norm, ""

            fb_up.click(_go_up, fb_path, [fb_list, fb_path, fb_selected])

            def _open_sel(selected, path, query):
                target = selected if selected and Path(selected).is_dir() else path
                choices, norm = _list_dir(target, query)
                return gr.update(choices=choices, value=None), norm, ""

            fb_open.click(
                _open_sel, [fb_selected, fb_path, fb_query], [fb_list, fb_path, fb_selected]
            )

            gr.Markdown(
                "### Load a training run → register a checkpoint\n"
                "A **run dir** is a `run_experiment` output folder (holds `lora_epoch_NNN/`, "
                "`latest.pth`, `args.json`, `grpo_config.json`). Scan it and register a checkpoint: "
                "a `lora_epoch_*` becomes a **LoRA** (pair it with any base model), a "
                "`merged.pth`/`best_model.pth` becomes a **Model**."
            )
            with gr.Row():
                run_dir_box = gr.Textbox(label="Run dir", scale=3)
                scan_btn = gr.Button("Scan run")
            run_scan_state = gr.State({})
            ckpt_dd = gr.Dropdown(label="Checkpoint", choices=[])
            with gr.Row():
                reg_name = gr.Textbox(label="Register as name", scale=3)
                reg_btn = gr.Button("Register checkpoint")
            run_status = gr.Textbox(label="", interactive=False)
            run_cfg = gr.JSON(label="grpo_config.json")

        # ---- workflow tabs ---------------------------------------------------------
        with gr.Tab("Train"):
            workflow_panel(wf("train_ranked_sft"), library0, library_state, asset_dropdowns)
            gr.Markdown("### Per-epoch metrics (from the latest train log)")
            refresh_btn = gr.Button("Refresh epoch table")
            ep_status = gr.Textbox(label="", interactive=False)
            ep_table = gr.Dataframe(headers=_EPOCH_HEADERS)
            refresh_btn.click(lambda: _epoch_table("train_ranked_sft"), None, [ep_table, ep_status])

        with gr.Tab("Evaluate"):
            gr.Markdown(_baseline_md(library0))
            with gr.Tab("Det avoidance (sc)"):
                ev = workflow_panel(
                    wf("eval_det_avoidance"), library0, library_state, asset_dropdowns
                )
                load_btn = gr.Button("Load summary.json")
                summ = gr.JSON(label="summary.json")

                def _load_summary(library, *flat, _ev=ev):
                    v = resolve_values(wf("eval_det_avoidance"), _ev["fields"], library, flat)
                    p = (wf("eval_det_avoidance").outputs or (lambda _: {}))(v).get("summary_json")
                    if p and Path(p).exists():
                        return json.loads(Path(p).read_text())
                    return {"error": f"not found: {p}"}

                load_btn.click(_load_summary, [library_state, *ev["flat"]], summ)
            with gr.Tab("Guided avoidance (policy)"):
                workflow_panel(
                    wf("eval_policy_avoidance"), library0, library_state, asset_dropdowns
                )
            with gr.Tab("L2 (valid_predictor)"):
                workflow_panel(wf("eval_l2"), library0, library_state, asset_dropdowns)
            with gr.Tab("Road border"):
                workflow_panel(wf("eval_border_distance"), library0, library_state, asset_dropdowns)
            with gr.Tab("Detailed metrics"):
                workflow_panel(
                    wf("eval_detailed_metrics"), library0, library_state, asset_dropdowns
                )

        with gr.Tab("Merge + Export"):
            with gr.Tab("Merge LoRA"):
                workflow_panel(wf("merge_lora"), library0, library_state, asset_dropdowns)
            with gr.Tab("Export ONNX"):
                workflow_panel(wf("torch2onnx"), library0, library_state, asset_dropdowns)

        with gr.Tab("PRiSM"):
            for k in ("disturb_and_replay", "viz_p4_recovery", "percentile_filter_perturbed"):
                with gr.Tab(wf(k).title.split(":")[0]):
                    workflow_panel(wf(k), library0, library_state, asset_dropdowns)

        def _viz_with_viewer(vwf):
            """A workflow panel plus a 'Load rendered outputs' WebM/PNG viewer."""
            vp = workflow_panel(vwf, library0, library_state, asset_dropdowns)
            show_btn = gr.Button("Load rendered outputs")
            vid = gr.Video(label="WebM")
            gallery = gr.Gallery(label="PNGs", columns=4, height=460)
            vmsg = gr.Textbox(label="", interactive=False)

            def _show(library, *flat, _wf=vwf, _vp=vp):
                v = resolve_values(_wf, _vp["fields"], library, flat)
                return _scan_outputs(v.get("output_dir", ""))

            show_btn.click(_show, [library_state, *vp["flat"]], [vid, gallery, vmsg])

        with gr.Tab("Reproducer / Viz"):
            with gr.Tab("Mine collisions"):
                workflow_panel(wf("mine_collisions"), library0, library_state, asset_dropdowns)
            with gr.Tab("Ghost A/B"):
                cl_check = gr.Checkbox(
                    value=False,
                    label="Closed-loop (re-inference each step). Unchecked = open-loop perfect-track.",
                )
                with gr.Group(visible=True) as open_grp:
                    _viz_with_viewer(wf("ghost_replay_openloop"))
                with gr.Group(visible=False) as closed_grp:
                    _viz_with_viewer(wf("compare_models_ghost"))
                cl_check.change(
                    lambda c: (gr.update(visible=not c), gr.update(visible=c)),
                    cl_check,
                    [open_grp, closed_grp],
                )
            with gr.Tab("Render NPZ dir"):
                _viz_with_viewer(wf("render_npz_dir"))

        with gr.Tab("Scene Editor"):
            sewf = wf("scene_branch_editor")
            gr.Markdown(f"**{sewf.title}** — {sewf.description} Runs in-process on its own port.")
            se_fields, se_flat = build_form(sewf, library0, asset_dropdowns)
            ed_port = gr.Number(value=default_editor_port, precision=0, label="Editor port")
            open_btn = gr.Button("▶ Open / Reload editor", variant="primary")
            se_status = gr.Textbox(label="Status", interactive=False)
            se_frame = gr.HTML()
            open_btn.click(
                lambda library, port, *flat: _open_editor(library, host, port, se_fields, *flat),
                [library_state, ed_port, *se_flat],
                [se_frame, se_status],
            )

        # ---- wire Workspace handlers (now that all dropdowns are collected) --------
        flat_dropdowns: list = []
        flat_meta: list = []  # (type, required) parallel to flat_dropdowns
        for t in P.LIST_TYPES:
            for dd, req in asset_dropdowns.get(t, []):
                flat_dropdowns.append(dd)
                flat_meta.append((t, req))

        def _dropdown_updates(library):
            return [gr.update(choices=_list_choices(library, t, req)) for (t, req) in flat_meta]

        def _remove_choices(library):
            out = []
            for t in P.LIST_TYPES:
                out += [f"{t}/{n}" for n in P.entry_names(library, t)]
            return out

        def _add(asset_type, name, selected, curdir, library):
            name = (name or "").strip()
            sel_path = (selected or "").strip() or (curdir or "").strip()
            if not name or not sel_path:
                return (
                    library,
                    _library_html(library),
                    "⚠ select a path in the browser and enter a name",
                    gr.update(),
                    *_dropdown_updates(library),
                )
            entry, note = _resolve_asset(asset_type, sel_path)
            entry["name"] = name
            library.setdefault(asset_type, []).append(entry)
            P.save_library(library)
            msg = f"added {asset_type}: {name}" + (f"  ({note})" if note else "")
            return (
                library,
                _library_html(library),
                msg,
                gr.update(choices=_remove_choices(library)),
                *_dropdown_updates(library),
            )

        add_outputs = [library_state, lib_view, ws_status, remove_dd, *flat_dropdowns]
        for btn, typ in (
            (add_model, "models"),
            (add_lora, "loras"),
            (add_policy, "policies"),
            (add_dataset, "datasets"),
            (add_reward, "reward_configs"),
            (add_map, "maps"),
            (add_run, "run_dirs"),
        ):
            btn.click(
                lambda name, sel, cur, lib, _t=typ: _add(_t, name, sel, cur, lib),
                [add_name, fb_selected, fb_path, library_state],
                add_outputs,
            )

        def _remove(sel, library):
            if sel and "/" in sel:
                t, n = sel.split("/", 1)
                library[t] = [e for e in library.get(t, []) if e.get("name") != n]
                P.save_library(library)
                msg = f"removed {sel}"
            else:
                msg = "nothing selected"
            return (
                library,
                _library_html(library),
                msg,
                gr.update(choices=_remove_choices(library)),
                *_dropdown_updates(library),
            )

        remove_btn.click(_remove, [remove_dd, library_state], add_outputs)

        def _set_scalar(key, val, library):
            library[key] = val
            P.save_library(library)
            return library, _library_html(library)

        ego_box.change(
            lambda v, lib: _set_scalar("ego_shape", v, lib),
            [ego_box, library_state],
            [library_state, lib_view],
        )
        out_box.change(
            lambda v, lib: _set_scalar("output_dir", v, lib),
            [out_box, library_state],
            [library_state, lib_view],
        )

        scan_btn.click(_scan_run, run_dir_box, [ckpt_dd, run_status, run_scan_state]).then(
            lambda st: st.get("cfg", {}), run_scan_state, run_cfg
        )

        def _register_ckpt(name, ckpt_label, scan, library):
            found = (scan or {}).get("found", {})
            if not ckpt_label or ckpt_label not in found:
                return (
                    library,
                    _library_html(library),
                    "⚠ scan a run and pick a checkpoint",
                    gr.update(),
                    *_dropdown_updates(library),
                )
            info = found[ckpt_label]
            nm = (name or ckpt_label).strip()
            # A lora_epoch checkpoint registers as a LoRA (combine with any base model);
            # a merged/best_model.pth registers as a Model.
            if info.get("lora_dir"):
                library.setdefault("loras", []).append({"name": nm, "path": info["lora_dir"]})
                kind = "LoRA"
            else:
                entry = {"name": nm, "path": info["path"]}
                if info.get("args_json"):
                    entry["args_json"] = info["args_json"]
                library.setdefault("models", []).append(entry)
                kind = "model"
            P.save_library(library)
            return (
                library,
                _library_html(library),
                f"registered {kind}: {nm}",
                gr.update(choices=_remove_choices(library)),
                *_dropdown_updates(library),
            )

        reg_btn.click(
            _register_ckpt, [reg_name, ckpt_dd, run_scan_state, library_state], add_outputs
        )

        # initialise the remove dropdown choices
        demo.load(lambda lib: gr.update(choices=_remove_choices(lib)), library_state, remove_dd)

    return demo


def main():
    ap = argparse.ArgumentParser(description="Autoresearch control panel")
    ap.add_argument("--port", type=int, default=7888)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--editor_port", type=int, default=7899)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    demo = build_app(host="localhost", default_editor_port=args.editor_port)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
