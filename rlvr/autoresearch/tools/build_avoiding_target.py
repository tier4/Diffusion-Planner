"""Deprecated alias for :mod:`rlvr.autoresearch.tools.build_repaired_targets`.

The tool was renamed because it repairs ANY rule violation (road-border
crossing, moving/static collision), not just "avoidance" — the misleading old
name implied avoidance-only. This shim is kept only so an in-flight workflow that
already baked ``python -m rlvr.autoresearch.tools.build_avoiding_target`` into a
subprocess command keeps resolving. New code must import
``build_repaired_targets`` directly; this module will be removed.
"""

from __future__ import annotations

import warnings

from rlvr.autoresearch.tools.build_repaired_targets import main

warnings.warn(
    "build_avoiding_target is renamed to build_repaired_targets; update imports.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    main()
