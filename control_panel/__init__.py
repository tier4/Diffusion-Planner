"""Unified Gradio control panel for the autoresearch workflows.

A thin launcher layer: it builds forms from a workflow registry, shells out to the
existing CLI tools (no training / inference / scoring is reimplemented here), streams
their logs, and chains outputs into the next step. See ``control_panel/README.md``.
"""
