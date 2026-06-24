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

NONE = "(none)"
ADD = "➕ Browse to add…"  # sentinel: selecting it opens the OS picker to register a new asset

_TYPE_ICON = {
    "models": "🧠",
    "loras": "🧩",
    "policies": "🛰",
    "reward_configs": "⚙️",
    "maps": "🗺",
    "route_datasets": "📁",
    "scene_datasets": "🎬",
    "run_dirs": "📦",
}


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
    if spec.auto:
        return "auto"
    if spec.derive_from:
        return "derive"
    if spec.shared == "ego_shape":
        return "ego"
    if spec.shared in P.LIST_TYPES:
        return "list"
    return "plain"


def _browse_button(textbox, mode: str):
    """Attach a 📂 button that opens the OS picker and fills ``textbox``."""
    b = gr.Button("📂", scale=1, min_width=44)
    b.click(lambda cur, _m=mode: _os_pick(_m, cur or "")[0] or cur, textbox, textbox)
    return b


def _list_choices(library: dict, asset_type: str, required: bool) -> list[str]:
    base = P.entry_names(library, asset_type)
    return ([] if required else [NONE]) + base + [ADD]


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
    dev_defaults = P.field_defaults().get(wf.key, {})  # gitignored testing pre-fills
    # Workflows with auto-derived outputs get a single "Run name" field; everything else
    # (the run folder + output files) is placed under <workspace output_dir>/<key>/<run>.
    if any(a.auto for a in wf.args):
        if wf.creates in ("scenes", "routes"):
            lbl = f"Dataset name → datasets/{wf.creates}/<name>/"
        else:
            lbl = f"Run name → runs/{wf.key}/<name>/"
        run_tb = gr.Textbox(value=wf.key, label=lbl)
        fields.append({"name": "__run__", "spec": None, "role": "runname", "comps": [run_tb]})
        flat.append(run_tb)

    def _list_widget(spec, scale=None):
        """One dropdown over the library for a shared-list field (+ an inline 'Browse to add')."""
        names = _list_choices(library, spec.shared, spec.required)
        base = P.entry_names(library, spec.shared)
        value = names[0] if (spec.required and base) else (NONE if not spec.required else None)
        icon = _TYPE_ICON.get(spec.shared, "")
        dd = gr.Dropdown(
            choices=names,
            value=value,
            label=f"{icon} {spec.label}" + (" *" if spec.required else ""),
            info=spec.help or None,
            scale=scale,
        )
        last = gr.State(value)  # last valid selection, for cancel-revert
        pick_mode = "dir" if spec.shared in ("policies", "run_dirs", "route_datasets") else "file"
        asset_dropdowns.setdefault(spec.shared, []).append((dd, spec.required, last, pick_mode))
        return dd

    specs = list(wf.args)
    i = 0
    while i < len(specs):
        spec = specs[i]
        role = _field_role(spec)
        nxt = specs[i + 1] if i + 1 < len(specs) else None
        # Group a model field with the LoRA that immediately follows it on one row.
        if (
            role == "list"
            and spec.shared == "models"
            and nxt is not None
            and _field_role(nxt) == "list"
            and nxt.shared == "loras"
        ):
            with gr.Row():
                dd_m = _list_widget(spec, scale=3)
                dd_l = _list_widget(nxt, scale=2)
            fields.append({"name": spec.name, "spec": spec, "role": "list", "comps": [dd_m]})
            flat.append(dd_m)
            fields.append({"name": nxt.name, "spec": nxt, "role": "list", "comps": [dd_l]})
            flat.append(dd_l)
            i += 2
            continue

        comps = []
        if role in ("hidden", "auto", "derive", "ego"):
            comps = []  # not rendered; resolved at Run
        elif role == "list":
            comps = [_list_widget(spec)]
        elif role == "plain" and spec.kind in ("file", "dir", "path"):
            with gr.Row():
                tb = _plain_widget(spec, default=dev_defaults.get(spec.name))
                _browse_button(tb, "dir" if spec.kind == "dir" else "file")
            comps = [tb]
        else:  # plain non-path (str/int/float/bool/choice)
            comps = [_plain_widget(spec)]
        fields.append({"name": spec.name, "spec": spec, "role": role, "comps": comps})
        flat.extend(comps)
        i += 1
    return fields, flat


def _ws_base(library: dict) -> str:
    """Base dir for auto outputs: the workspace root (fallback: legacy output_dir)."""
    return library.get("workspace_root") or library.get("output_dir", "")


def _run_dir(library: dict, wf: W.Workflow, run: str) -> Path | None:
    """The folder a run writes into, per wf.creates: datasets/scenes|routes/<run> or runs/<key>/<run>."""
    base = _ws_base(library)
    if not base:
        return None
    run = run or "run"
    if wf.creates == "scenes":
        return Path(base) / "datasets" / "scenes" / run
    if wf.creates == "routes":
        return Path(base) / "datasets" / "routes" / run
    return Path(base) / "runs" / wf.key / run


def _auto_path(library: dict, wf: W.Workflow, run: str, spec: W.ArgSpec) -> str:
    """Derive an output path under the run folder.

    For creates='scenes', a `file:*.json` (the scene list) is placed at the scenes/ level as
    <run>.json (sibling of the npz dir) so the workspace scanner picks it up as a dataset.
    """
    rd = _run_dir(library, wf, run)
    if rd is None:
        return ""
    kind, _, rest = spec.auto.partition(":")
    if wf.creates == "scenes" and kind == "file" and rest.endswith(".json"):
        return str(rd.parent / f"{run or 'run'}.json")
    if kind == "dir":
        return str(rd / rest) if rest else str(rd)
    if kind == "file":
        return str(rd / rest)
    raise ValueError(f"{spec.name}: bad auto spec {spec.auto!r}")


def resolve_values(
    wf: W.Workflow, fields: list, library: dict, flat_values: tuple, make_dirs: bool = False
) -> dict:
    """Reconstruct the CLI values dict from the flat form values + the live library.

    Auto-output fields are placed under <workspace output_dir>/<wf.key>/<run name>. When
    make_dirs is True (at Run, not Preview) the run folder is created so file outputs land.
    """
    it = iter(flat_values)
    values: dict = {}
    model_entries: dict = {}
    run_name = ""
    for f in fields:
        spec, role, name = f["spec"], f["role"], f["name"]
        vals = [next(it) for _ in f["comps"]]
        if role == "runname":
            run_name = (vals[0] or "run").strip()
        elif role == "hidden":
            values[name] = spec.default
        elif role == "ego":
            values[name] = library.get("ego_shape", "")
        elif role in ("auto", "derive"):
            pass  # filled in the second pass
        elif role == "list":
            sel = vals[0]
            if sel in (NONE, ADD, None, ""):
                values[name] = ""
                entry = {}
            else:
                entry = P.find_entry(library, spec.shared, sel) or {}
                values[name] = entry.get("path", "")
            if spec.shared == "models":
                model_entries[name] = entry
        else:
            values[name] = _coerce(spec, vals[0])

    run_dir = _run_dir(library, wf, run_name)
    if make_dirs and run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
    for f in fields:
        spec = f["spec"]
        if f["role"] == "auto":
            values[f["name"]] = _auto_path(library, wf, run_name, spec)
        elif f["role"] == "derive":
            entry = model_entries.get(spec.derive_from, {})
            values[f["name"]] = entry.get(spec.derive_field, "") or ""
    return values


# --------------------------------------------------------------------------------------
# Run / preview / stop / attach handlers
# --------------------------------------------------------------------------------------
def _err(msg: str) -> str:
    return f"<span style='color:#d33'>⚠ {msg}</span>"


def _needs_output_dir(wf, library) -> str:
    """Red error if the workflow auto-derives outputs but no workspace root / output dir is set."""
    if any(a.auto for a in wf.args) and not _ws_base(library):
        return _err("Set the Workspace root in the Workspace tab first (outputs go there).")
    return ""


def _run_handler(wf, fields):
    def run(library, *flat):
        guard = _needs_output_dir(wf, library)
        if guard:
            yield "", "not started", guard, None
            return
        values = resolve_values(wf, fields, library, flat, make_dirs=True)
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
        guard = _needs_output_dir(wf, library)
        if guard:
            return guard
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

    run_evt = run_btn.click(
        _run_handler(wf, fields), [library_state, *flat], [log, status, err, job_state]
    )
    stop_btn.click(_stop_handler(), job_state, status)
    prev_btn.click(_preview_handler(wf, fields), [library_state, *flat], log)
    attach_btn.click(_attach_handler(wf.key), None, [log, status, job_state])
    return {
        "fields": fields,
        "flat": flat,
        "log": log,
        "status": status,
        "run_evt": run_evt,
        "wf": wf,
    }


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


_TK_PICK = (
    "import sys, tkinter as tk\n"
    "from tkinter import filedialog\n"
    "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
    "mode, start = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else '')\n"
    "p = filedialog.askdirectory(initialdir=start or None) if mode == 'dir' "
    "else filedialog.askopenfilename(initialdir=start or None)\n"
    "print(p or '')\n"
)


def _os_pick(mode: str, start: str = "") -> tuple[str, str]:
    """Open the native OS file/folder picker on the server's desktop. Returns (path, note).

    Prefers zenity (GTK), falls back to a tkinter subprocess. Runs the dialog in a short-lived
    subprocess so it never blocks/wedges the Gradio worker thread. Empty path = cancelled.
    """
    import shutil
    import subprocess
    import sys

    zenity = shutil.which("zenity")
    try:
        if zenity:
            cmd = [zenity, "--file-selection", "--title", f"Select {mode}"]
            if mode == "dir":
                cmd.append("--directory")
            if start:
                cmd += ["--filename", start.rstrip("/") + "/"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            return (r.stdout.strip(), "") if r.returncode == 0 else ("", "cancelled")
        r = subprocess.run(
            [sys.executable, "-c", _TK_PICK, mode, start],
            capture_output=True,
            text=True,
            timeout=600,
        )
        path = r.stdout.strip()
        return (path, "") if path else ("", "cancelled")
    except Exception as e:  # noqa: BLE001 - surface picker/display problems to the UI
        return "", f"picker unavailable ({e}); paste the path instead"


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
# Scene Editor (subprocess on its own port + iframe — needs the lanelet env)
# --------------------------------------------------------------------------------------
def _open_editor(library, host, editor_port, fields, *flat):
    sewf = W.get_workflow("scene_branch_editor")
    values = resolve_values(sewf, fields, library, flat)
    if not values.get("npz_dir"):
        return "", "⚠ Select a Replay NPZ dir (route) before opening the editor."
    values["port"] = int(editor_port)
    # Pre-point the editor's export/save dirs into the workspace: Export NPZs = contiguous
    # (→ datasets/routes), Save Scene + Guided Traj = individual scenes (→ datasets/scenes).
    ws = _ws_base(library)
    if ws:
        values["export_dir"] = str(Path(ws) / "datasets" / "routes" / "editor_export")
        values["rsft_dir"] = str(Path(ws) / "datasets" / "scenes" / "editor_curated")
    # Stop a previous editor (interactive server, restartable) before relaunching.
    prev = R.latest_job("scene_branch_editor")
    if prev is not None and R.is_alive(prev.pid):
        R.stop(prev)
    try:
        job = R.launch(sewf, values)
    except ValueError as e:
        return "", _err(str(e))
    url = f"http://{host or 'localhost'}:{int(editor_port)}"
    html = (
        f'<p><a href="{url}" target="_blank">Open in new tab ↗</a> '
        f"(PID {job.pid}; give it ~10s to boot, then it appears below)</p>"
        f'<iframe src="{url}" width="100%" height="900" '
        'style="border:1px solid #ccc;border-radius:6px"></iframe>'
    )
    return html, f"editor launching on {url} (log: {job.logfile})"


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
        creating_panels: list = []  # panels whose tool writes a dataset → auto-rescan on finish

        # ---- Workspace -------------------------------------------------------------
        with gr.Tab("Workspace"):
            gr.Markdown(
                "### Workspace — point at your workspace root and **Scan** to load all assets"
            )
            with gr.Row():
                ws_root_box = gr.Textbox(
                    value=library0.get("workspace_root", ""), label="Workspace root", scale=4
                )
                ws_browse_btn = gr.Button("📁", scale=1, min_width=44)
                ws_create_btn = gr.Button("🆕 Create folders", scale=2)
                scan_btn2 = gr.Button("🔄 Scan workspace", variant="primary", scale=2)

            gr.Markdown("### Or register a one-off asset outside the workspace")
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        pick_file_btn = gr.Button("📂 Browse file…")
                        pick_dir_btn = gr.Button("📁 Browse folder…")
                    picked = gr.Textbox(label="Selected path (Browse, or paste/type)")
                with gr.Column(scale=1):
                    add_name = gr.Textbox(label="Name for the asset")
                    add_model = gr.Button("Add as Model", variant="primary")
                    add_lora = gr.Button("Add as LoRA")
                    add_policy = gr.Button("Add as Guidance policy")
                    add_scene_ds = gr.Button("Add as Scene dataset")
                    add_route_ds = gr.Button("Add as Route dataset")
                    add_reward = gr.Button("Add as Reward config")
                    add_map = gr.Button("Add as Map")
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

            # native OS file picker
            def _pick(mode, current, _start=library0.get("ssd_root") or ""):
                path, note = _os_pick(mode, _start)
                return (path, f"selected {path}") if path else (current, note or "cancelled")

            pick_file_btn.click(lambda cur: _pick("file", cur), picked, [picked, ws_status])
            pick_dir_btn.click(lambda cur: _pick("dir", cur), picked, [picked, ws_status])

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
            with gr.Tab("RSFT"):
                workflow_panel(wf("train_ranked_sft"), library0, library_state, asset_dropdowns)
                gr.Markdown("### Per-epoch metrics (from the latest train log)")
                refresh_btn = gr.Button("Refresh epoch table")
                ep_status = gr.Textbox(label="", interactive=False)
                ep_table = gr.Dataframe(headers=_EPOCH_HEADERS)
                refresh_btn.click(
                    lambda: _epoch_table("train_ranked_sft"), None, [ep_table, ep_status]
                )
            with gr.Tab("PRiSM"):
                gr.Markdown(
                    "Perturbation-Recovery Self-Mining: disturb warm scenes → rank K=N → filter → "
                    "RSFT-train. Run the steps in order."
                )
                for k in ("disturb_and_replay", "viz_p4_recovery", "percentile_filter_perturbed"):
                    with gr.Tab(k.replace("_perturbed", "").replace("_", " ")):
                        p = workflow_panel(wf(k), library0, library_state, asset_dropdowns)
                        if wf(k).creates:
                            creating_panels.append(p)

        with gr.Tab("Evaluate"):
            with gr.Tab("Metrics"):
                use_guid = gr.Checkbox(
                    value=False,
                    label="Use guidance policy (guided eval) — unchecked = plain deterministic",
                )
                with gr.Group(visible=True) as det_grp:
                    ev = workflow_panel(
                        wf("eval_full_metrics"), library0, library_state, asset_dropdowns
                    )
                    with gr.Row():
                        load_btn = gr.Button("Load summary.json")
                        viz_btn = gr.Button("Load scene viz (tick Render first)")
                    summ = gr.JSON(label="summary.json")
                    ev_gallery = gr.Gallery(label="Rendered scenes", columns=3, height=520)

                    def _eval_out(library, flat):
                        v = resolve_values(wf("eval_full_metrics"), ev["fields"], library, flat)
                        return v.get("output_dir", "")

                    def _load_summary(library, *flat):
                        out = _eval_out(library, flat)
                        p = Path(out) / "summary.json" if out else None
                        if p and p.exists():
                            return json.loads(p.read_text())
                        return {"error": f"not found: {p} (run the eval first)"}

                    def _load_viz(library, *flat):
                        out = _eval_out(library, flat)
                        return _scan_outputs(out)[1] if out else []

                    load_btn.click(_load_summary, [library_state, *ev["flat"]], summ)
                    viz_btn.click(_load_viz, [library_state, *ev["flat"]], ev_gallery)
                with gr.Group(visible=False) as guid_grp:
                    workflow_panel(
                        wf("eval_policy_avoidance"), library0, library_state, asset_dropdowns
                    )
                use_guid.change(
                    lambda c: (gr.update(visible=not c), gr.update(visible=c)),
                    use_guid,
                    [det_grp, guid_grp],
                )
            with gr.Tab("L2 loss"):
                workflow_panel(wf("eval_l2"), library0, library_state, asset_dropdowns)

        with gr.Tab("Merge + Export"):
            with gr.Tab("Merge LoRA"):
                workflow_panel(wf("merge_lora"), library0, library_state, asset_dropdowns)
            with gr.Tab("Export ONNX"):
                workflow_panel(wf("torch2onnx"), library0, library_state, asset_dropdowns)

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
            gr.Markdown(f"**{sewf.title}** — {sewf.description}")
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
            for dd, req, *_ in asset_dropdowns.get(t, []):
                flat_dropdowns.append(dd)
                flat_meta.append((t, req))

        def _dropdown_updates(library):
            return [gr.update(choices=_list_choices(library, t, req)) for (t, req) in flat_meta]

        def _remove_choices(library):
            out = []
            for t in P.LIST_TYPES:
                out += [f"{t}/{n}" for n in P.entry_names(library, t)]
            return out

        def _add(asset_type, name, selected, library):
            name = (name or "").strip()
            sel_path = (selected or "").strip()
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
            (add_scene_ds, "scene_datasets"),
            (add_route_ds, "route_datasets"),
            (add_reward, "reward_configs"),
            (add_map, "maps"),
        ):
            btn.click(
                lambda name, sel, lib, _t=typ: _add(_t, name, sel, lib),
                [add_name, picked, library_state],
                add_outputs,
            )

        # Workspace root + Scan: rebuild the whole library from the folder layout.
        ws_browse_btn.click(
            lambda cur: _os_pick("dir", cur or "")[0] or cur, ws_root_box, ws_root_box
        )

        def _scan_ws(root, library):
            if not root or not Path(root).is_dir():
                return (
                    library,
                    _library_html(library),
                    f"⚠ not a folder: {root}",
                    gr.update(),
                    *_dropdown_updates(library),
                )
            scanned = P.scan_workspace(root)
            # keep scalars the user may have set
            for k in ("ego_shape", "output_dir"):
                if library.get(k):
                    scanned[k] = library[k]
            P.save_library(scanned)
            counts = ", ".join(
                f"{t}:{len(scanned.get(t, []))}" for t in P.LIST_TYPES if scanned.get(t)
            )
            return (
                scanned,
                _library_html(scanned),
                f"scanned {root} → {counts or 'nothing found'}",
                gr.update(choices=_remove_choices(scanned)),
                *_dropdown_updates(scanned),
            )

        scan_btn2.click(_scan_ws, [ws_root_box, library_state], add_outputs)
        ws_root_box.change(
            lambda v, lib: _set_scalar("workspace_root", v, lib),
            [ws_root_box, library_state],
            [library_state, lib_view],
        )

        def _create_ws(root):
            if not (root or "").strip():
                return "⚠ enter a path for the new workspace first"
            try:
                P.create_workspace(root)
            except OSError as e:
                return _err(f"could not create workspace: {e}")
            subs = ", ".join(P.WORKSPACE_DIRS.values())
            return f"created workspace at {root} ({subs}, runs) — drop assets in & Scan"

        ws_create_btn.click(_create_ws, ws_root_box, ws_status)

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

        # ---- inline "Browse to add…" on every asset dropdown ----------------------
        def _auto_name(library, asset_type, path):
            stem = Path(path.rstrip("/")).stem or Path(path.rstrip("/")).name or asset_type
            existing = set(P.entry_names(library, asset_type))
            name, n = stem, 2
            while name in existing:
                name, n = f"{stem}-{n}", n + 1
            return name

        def _make_add_handler(asset_type, my_idx, type_dds, type_reqs, pick_mode):
            def on_change(sel, last, library):
                # type choices refresh (no value) for every dropdown of this type
                def type_ups(value_for_me=None):
                    ups = []
                    for i, req in enumerate(type_reqs):
                        ch = _list_choices(library, asset_type, req)
                        ups.append(
                            gr.update(choices=ch, value=value_for_me)
                            if (i == my_idx and value_for_me is not None)
                            else gr.update(choices=ch)
                        )
                    return ups

                if sel != ADD:  # normal selection (or the revert echo) — just record it
                    return (library, sel, gr.update(), gr.update(), *type_ups())
                start = library.get("workspace_root") or library.get("ssd_root") or ""
                path, _note = _os_pick(pick_mode, start)
                if not path:  # cancelled → revert this dropdown to its last valid value
                    ups = type_ups()
                    ups[my_idx] = gr.update(
                        choices=_list_choices(library, asset_type, type_reqs[my_idx]), value=last
                    )
                    return (library, last, gr.update(), gr.update(), *ups)
                entry, _n = _resolve_asset(asset_type, path)
                entry["name"] = _auto_name(library, asset_type, path)
                library.setdefault(asset_type, []).append(entry)
                P.save_library(library)
                new = entry["name"]
                return (
                    library,
                    new,
                    _library_html(library),
                    gr.update(choices=_remove_choices(library)),
                    *type_ups(value_for_me=new),
                )

            return on_change

        for t in P.LIST_TYPES:
            entries = asset_dropdowns.get(t, [])
            type_dds = [dd for dd, *_ in entries]
            type_reqs = [req for _, req, *_ in entries]
            for idx, (dd, _req, last, pick_mode) in enumerate(entries):
                dd.change(
                    _make_add_handler(t, idx, type_dds, type_reqs, pick_mode),
                    inputs=[dd, last, library_state],
                    outputs=[library_state, last, lib_view, remove_dd, *type_dds],
                )

        # ---- auto-rescan: a tool that wrote a dataset → merge new workspace datasets in ----
        def _merge_workspace_datasets(library):
            """Add any workspace scene/route datasets not already in the library (keep one-offs)."""
            root = library.get("workspace_root")
            if root:
                scanned = P.scan_workspace(root)
                for t in ("scene_datasets", "route_datasets"):
                    have = {e.get("path") for e in library.get(t, [])}
                    for e in scanned.get(t, []):
                        if e.get("path") not in have:
                            library.setdefault(t, []).append(e)
                P.save_library(library)
            return (
                library,
                _library_html(library),
                gr.update(choices=_remove_choices(library)),
                *_dropdown_updates(library),
            )

        _rescan_outputs = [library_state, lib_view, remove_dd, *flat_dropdowns]
        for p in creating_panels:
            p["run_evt"].then(_merge_workspace_datasets, library_state, _rescan_outputs)

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
