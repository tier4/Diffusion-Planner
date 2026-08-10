### Task 7 Report: Multi-Human Location Report

**Status:** DONE

**Commits:** a3d90165

**Files created:**
- `human_match_prototype/multi_human_report.py` -- `diagnose_mismatch()`, `render_location_report()`, CLI entry point
- `human_match_prototype/tests/test_multi_human_report.py` -- 4 tests for diagnosis logic

**Implementation details:**
- `diagnose_mismatch(row)` applies thresholds in order: no_mismatch (< 0.5), insufficient_data (n_humans < 5), unusual_test_human (coverage >= 0.5), planner_deficiency (fallthrough)
- `render_location_report()` produces self-contained HTML with metrics table and base64-embedded BEV overlay PNGs
- CLI: `python -m human_match_prototype.multi_human_report --multi_csv CSV --output_dir DIR [--top_k 20]`
- Follows existing `cluster_report.py` patterns (inline CSS, b64 images, matplotlib Agg backend)

**Test summary:** `.venv/bin/python -m pytest human_match_prototype/tests/test_multi_human_report.py -v` -- 4 passed
Full suite: 36 passed, 0 failed

**Concerns:** None

---

**Review fix: render_location_report empty-rows guard + remove unused matplotlib import**

- Added early return in `render_location_report()` when `rows` is empty -- writes a "No mismatches found" HTML page instead of crashing with IndexError on `rows[0]`
- Removed unused `import matplotlib` / `matplotlib.use("Agg")` (matplotlib was never used in the module)
- Added test `test_render_location_report_empty_rows` to cover the empty-rows path
- Full suite: 37 passed, 0 failed
