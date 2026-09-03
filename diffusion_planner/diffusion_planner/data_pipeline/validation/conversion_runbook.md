# Conversion runbook: npz corpus → versioned tar shards

This is an operational procedure, not code. It walks a second operator through converting a
large corpus of per-frame `.npz` files plus JSON sidecars into a versioned shard dataset using
the `pack_shards` / `validation` CLIs in this package, on a machine whose filesystem is shared
with other running jobs. It assumes no memory of any prior conversation — every command below
was run against the shipped CLIs' `--help` output before being written down here, and every
flag used exists exactly as spelled.

Every path, count, and identifier in this document is a placeholder you fill in for your own
corpus and host. Nothing here should be run against real data until you have read the whole
document once, decided on values for every variable in "0. Before you start", and confirmed the
two traps below with your own setup.

## The two easiest ways to lose the whole run

1. **The partition regex must never end in a `.npz` anchor.** Internally, every tool strips the
   `.npz` suffix from a file's path before matching it against your `--partition-regex`. A
   pattern ending in `\.npz$` therefore matches nothing, and every single file raises a planning
   error. Use a regex whose capture group is the *containing directory*, such as:

   ```
   ^(?P<partition>.+)/[^/]+$
   ```

   This matches everything up to the final `/` — the frame file's own name — and leaves the
   directory it lives in as the partition. If your tree's directory depth is uniform you may be
   able to use `--partition-depth N` instead of a regex; if depths vary (some sources nest
   deeper than others) a regex is required, because a depth rule cannot express "one partition
   per leaf directory" across mixed depths.

2. **Pack the production corpus in batches with chained `--base` tags — never as one giant
   invocation.** A single `pack` call publishes nothing until *every* partition in it has
   finished building. On a corpus with many thousands of partitions, a crash near the end of a
   single giant invocation discards the entire run's work, published or not. Packing in batches
   of a few hundred to a thousand partitions each, with each batch's `--base` set to the
   previous batch's `--tag` (and the first batch's `--base` set to `none`), bounds a crash to at
   most one batch, and makes it possible to compare actual progress against your calibration
   projection partway through and course-correct — which is the entire reason Phase 3 below is
   structured the way it is.

A third trap worth flagging even though it is less obvious: the collision check that guards
against two different partitions hashing to the same short output-path id only ever looks at the
partitions given to *one* `pack` invocation. If you batch the production pack (as you must, per
trap 2), that per-invocation check can never see collisions *across* batches. Section 2 below
runs that check once, up front, over every partition in the corpus, specifically so batching
cannot silently skip it.

---

## 0. Before you start — set these first

Everything below refers to these shell variables. Set real values for your corpus and host
before running anything; the values shown are illustrative, not defaults to keep.

```bash
# --- corpus location ---
export SOURCE=/path/to/source/npz/tree        # read-only source tree, on a mount local to the packing host
export DEST=/path/to/shared/dataset/root      # destination dataset root, on the shared filesystem other jobs also read
export SCRATCH=/path/to/local/scratch         # packing host's OWN local disk — never the shared filesystem

# --- the path list you were handed ---
export PATH_LIST_IN=/path/to/original_path_list.json   # JSON list of npz paths, as given to you
export OLD_PREFIX=/prefix/as/seen/by/the/list           # prefix under which PATH_LIST_IN's entries are absolute
export NEW_PREFIX="$SOURCE"                             # the same tree's prefix on THIS host; must equal $SOURCE
export PATH_LIST="$SCRATCH/path_list.rewritten.json"    # prepare_path_list writes the rewritten list here

# --- partitioning ---
export PARTITION_REGEX='^(?P<partition>.+)/[^/]+$'      # partition = containing directory; see trap 1 above

# --- the shape you were told (or independently derived) this corpus should have ---
export EXPECTED_ENTRIES=0        # path-list entry count you were told to expect
export EXPECTED_SAMPLES=0        # total .npz count `inspect` should report under $SOURCE
export EXPECTED_PARTITIONS=0     # partition count `inspect` should report under $PARTITION_REGEX

# --- calibration (Phases 1-2) ---
export CALIB_SAMPLES=15000       # order-of-magnitude size for a calibration slice; tune to taste
export CANDIDATE_WORKERS="8 16 32"   # every worker count you intend to justify by measurement

# --- production pack (Phase 3) ---
export WORKERS=0                 # fill in only after Phase 2 justifies a number — do not guess ahead
export ETA=0                     # the η_W Phase 2 measured for THAT worker count — not any other candidate's
export BATCH_SIZE=750            # partitions per batch; keep it in the few-hundred-to-a-thousand range
export TAG_PREFIX=prod           # batch tags become ${TAG_PREFIX}-batch-0000, -0001, ...
export FINAL_TAG=""              # set once you know which batch tag you will promote after acceptance

# --- acceptance (Phase 4) ---
export SAMPLE_SEED=20260903      # any fixed integer; record it, do not change it once chosen
export MIN_BITEXACT_KEYS=1000    # spec floor for the bit-exact sample; raise if you want more confidence

# --- host safety ---
export FREE_SPACE_FLOOR_GB=0     # agreed with whoever owns the shared filesystem; fill in before Phase 3
```

Also settle, in writing, with the people who own the shared filesystem and any other jobs that
read through it, before Phase 2:

- the free-space floor above,
- who is named to watch other jobs' throughput during the first parallel-worker measurement,
- confirmation that the source tree will not change for the duration of the run (every check in
  this procedure is local to one partition at one point in time — nothing here re-checks a
  partition once it is done, and nothing here notices a change to a partition that lands after
  its own task started but before the whole corpus is scanned).

All commands below are run from the package directory:

```bash
cd diffusion_planner
```

---

## 1. Environment

Install the toolchain on the packing host's **own local disk** — never on the shared filesystem
that also hosts `$DEST`. A user-local `uv` and Python 3.10 are what this needs.

```bash
uv sync
```

**Budget local disk for the whole project, not just the packer.** This repository's
`pyproject.toml` has a single flat dependency list — no extras, no dependency groups to opt out
of — so the plain `uv sync` above installs everything the project depends on, `torch`,
`pytorch-lightning`, and `onnxruntime-gpu` (with their CUDA wheels) included. There is currently
no scoped, pack-only install target in this repo; size the packing host's local disk for the
full install, not for the handful of packages the pack path actually touches.

That said, the pack path itself stays small at runtime: `pack_shards` and `pack_bench` only ever
import `duckdb`, `numpy`, `pyarrow`, `safetensors`, and `zstandard`. Nothing about running the
pack requires a GPU on this host — the CUDA-capable packages installed above simply sit unused
for this part of the procedure.

Before touching any real data, run the existing test suite on this host, with this checkout, and
confirm it is green:

```bash
uv run python -m pytest tests/ -v
```

Record the pass/fail count as your baseline for this host. If any `tests/test_dp_*.py` test
fails, stop and resolve it before proceeding — every later step in this document assumes those
tests pass here.

One note: the loader-smoke step in Phase 4 exercises the DDP loader, which does depend on
`torch`. Since the `uv sync` above already installs it, that step needs no separate environment
by default. The only case where it would is if the packing host does not actually carry this
repo's own `uv sync` output — for instance, a deliberately trimmed environment you assembled by
hand outside this repo's dependency list, to avoid the disk cost noted above. In that case, run
the loader-smoke step from wherever your project's normal environment is available and can read
`$DEST` — it does not need to run from the packing host.

Renice every packing process (`nice -n 19` or your platform's equivalent) for the whole
operation; the shared filesystem's read bandwidth and metadata rate are the resource other jobs
are competing with you for, not CPU.

---

## 2. Preflight

Two checks, both before any packing starts.

### 2.1 Rewrite and validate the path list

The path list you were handed is written with paths absolute under whatever prefix the system
that produced it uses — not this host's prefix for the same tree. Rewrite it onto this host's
prefix, into a new file on local disk; the original is never modified:

```bash
uv run python -m diffusion_planner.data_pipeline.validation.prepare_path_list \
  --in "$PATH_LIST_IN" \
  --out "$PATH_LIST" \
  --old-prefix "$OLD_PREFIX" \
  --new-prefix "$NEW_PREFIX"
```

This prints a small JSON report — `entries`, `rewritten`, `sha256`, `out` — to stdout.
**Record all four fields; this is what makes the published dataset traceable to an exact
input.** `entries` and `rewritten` should be equal (the rewrite aborts on the first invalid entry
rather than silently dropping bad ones, so a smaller `rewritten` count cannot appear on its own).
If you were given `$EXPECTED_ENTRIES` ahead of time, confirm `entries` matches it now.

The rewrite only checks that every entry is a `.npz` path that exists under `--new-prefix`, with
no duplicates after rewriting. It does **not** check that a sidecar JSON exists for each entry,
and it does not check the count against anything external — that is what the next step is for.

### 2.2 Confirm the partition rule against the real tree, and the pid-collision preflight

```bash
uv run python -m diffusion_planner.data_pipeline.pack_shards inspect \
  --source "$SOURCE" \
  --partition-regex "$PARTITION_REGEX" \
  > "$SCRATCH/inspect_report.txt"
```

`inspect` walks the whole `--source` tree directly — it takes no `--path-list`, so it counts
every `.npz` file under `$SOURCE`, independent of what is named in `$PATH_LIST`. Open the report
and read:

```
npz files: <N>
...
partitions (<M>):
  <partition id>: <count>
  ...
missing sidecars: <K>; non-sidecar jsons: <...>
```

- Compare `<N>` against `$EXPECTED_SAMPLES`, and `<M>` against `$EXPECTED_PARTITIONS`. A mismatch
  in `<N>` usually means `$SOURCE` and the path list disagree about which files exist; a mismatch
  in `<M>`, or any `OUTSIDE RULE (...)` line in the report, means the regex does not fit this
  tree's actual directory shape — fix the regex (never the tool), then re-run this step.
- Note `<K>` (missing sidecars). This is the same count `--require-sidecars` enforces at publish
  time in Phase 3; a large number here is worth investigating now, while it is still cheap.

Next, turn that report into a clean partition table and use it to run the pid-collision check
over **every** partition in the corpus, once, up front — not per-batch. This matters because the
same check that `pack` runs automatically before it builds anything only ever looks at the
partitions given to that one invocation; since Phase 3 packs in batches (each invocation seeing
only a few hundred partitions), the automatic check can never see a collision between a partition
in batch 3 and one in batch 40. Running it here, over the whole corpus, in one pass, is the only
way to actually cover the whole corpus.

Save this as `$SCRATCH/preflight.py`:

```python
"""One-off preflight: parse an `inspect` report into a clean partition table, and check that
`pid_of()` (the function that derives each partition's short output-path id) is injective over
every partition in the corpus — not just over one batch's worth."""
import os
import re
from pathlib import Path

from diffusion_planner.data_pipeline.partition import pid_of

report = Path(os.environ["INSPECT_REPORT"]).read_text().splitlines()
start = next(i for i, l in enumerate(report) if l.startswith("partitions ("))
rows = []
for line in report[start + 1:]:
    m = re.match(r"^  (.+): (\d+)$", line)
    if not m:
        break
    rows.append((m.group(1), int(m.group(2))))

Path(os.environ["PARTITIONS_TSV"]).write_text(
    "\n".join(f"{pid}\t{n}" for pid, n in rows) + "\n"
)

seen, collisions = {}, []
for pid, _n in rows:
    h = pid_of(pid)
    if h in seen and seen[h] != pid:
        collisions.append((seen[h], pid, h))
    seen[h] = pid

print(f"{len(rows)} partitions, {len(seen)} distinct pid()s")
if collisions:
    print("COLLISION(S) FOUND — do not pack until this is resolved:")
    for a, b, h in collisions:
        print(f"  {a!r} and {b!r} both hash to {h!r}")
    raise SystemExit(1)
print("pid_of is injective over every partition in the corpus")
```

Run it:

```bash
export INSPECT_REPORT="$SCRATCH/inspect_report.txt"
export PARTITIONS_TSV="$SCRATCH/partitions.tsv"
uv run python "$SCRATCH/preflight.py"
```

Do not proceed past a collision. It means two different partitions would overwrite each other's
shards and manifests on disk; the only remedies are choosing a different partition rule, or
excluding one of the colliding partitions, and either needs a human decision, not an override.

---

## 3. Phase 1 — serial calibration

Goal: measure the single-worker packing rate (`r1`, in samples/second) and the output/input size
ratio, using only the plain serial path (`--workers` absent or `1`) — the exact path every
existing test in the suite already exercises, so this phase needs nothing new to trust.

### 3.1 Build a stratified, previously-unread slice

There is no shipped "pick a stratified sample of partitions" command, so build the slice with a
short, reusable script. It groups partitions by their top-level source directory (the first path
component of the partition id) and round-robins across those groups until the slice reaches your
target sample count — an adequate approximation of "matched by source mix" for calibration
purposes. It also tracks which partitions have already been used, so that repeated calls (Phase 1
now, Phase 2's fresh slice and each additional worker-count candidate later) never draw from the
same partition twice — the closest approximation to "unread" available without root access to
drop the page cache.

Save as `$SCRATCH/select_slice.py`:

```python
"""Pick a stratified, not-yet-used slice of partitions and the path-list subset for exactly
those partitions. Reads its parameters from the environment so the same script can be re-run for
every slice you need across Phases 1 and 2, each time with different SLICE_* / OUT_* values."""
import json
import os
import re
from collections import defaultdict
from pathlib import Path

partitions_tsv = Path(os.environ["PARTITIONS_TSV"])
used_path = Path(os.environ["USED_PARTITIONS"])
target_samples = int(os.environ["SLICE_TARGET_SAMPLES"])
out_partitions = Path(os.environ["SLICE_PARTITIONS_OUT"])
out_path_list = Path(os.environ["SLICE_PATH_LIST_OUT"])
full_path_list = Path(os.environ["PATH_LIST"])
source = os.environ["SOURCE"].rstrip("/")
partition_regex = re.compile(os.environ["PARTITION_REGEX"])

counts = {}
for line in partitions_tsv.read_text().splitlines():
    pid, n = line.split("\t")
    counts[pid] = int(n)

used = set(used_path.read_text().split()) if used_path.exists() else set()

groups = defaultdict(list)
for pid, n in counts.items():
    if pid not in used:
        groups[pid.split("/", 1)[0]].append((pid, n))
for g in groups.values():
    g.sort()

selected, total = [], 0
iters = {top: iter(g) for top, g in groups.items()}
while total < target_samples and iters:
    for top in list(iters):
        try:
            pid, n = next(iters[top])
        except StopIteration:
            del iters[top]
            continue
        selected.append(pid)
        total += n
    if total >= target_samples:
        break

out_partitions.write_text("\n".join(selected) + "\n")
used_path.write_text("\n".join(sorted(used | set(selected))) + "\n")

selected_set = set(selected)
entries = json.loads(full_path_list.read_text())
prefix = source + "/"
sliced = []
for e in entries:
    assert e.startswith(prefix), f"path list entry not under $SOURCE: {e}"
    key = e[len(prefix):-4]  # strip source prefix and the trailing ".npz"
    pid = partition_regex.match(key).group("partition")
    if pid in selected_set:
        sliced.append(e)
out_path_list.write_text(json.dumps(sliced))

print(
    f"selected {len(selected)} partitions, {total} samples (target {target_samples}); "
    f"{len(sliced)} path-list entries -> {out_path_list}"
)
```

Run it for Phase 1:

```bash
export USED_PARTITIONS="$SCRATCH/used_partitions.txt"
: > "$USED_PARTITIONS"   # empty — this is the very first slice drawn

export SLICE_TARGET_SAMPLES="$CALIB_SAMPLES"
export SLICE_PARTITIONS_OUT="$SCRATCH/phase1.partitions.txt"
export SLICE_PATH_LIST_OUT="$SCRATCH/phase1.path_list.json"
uv run python "$SCRATCH/select_slice.py"
```

### 3.2 Measure input bytes for the slice

```bash
uv run python - <<PY
import json, os
from pathlib import Path

entries = json.loads(Path(os.environ["SLICE_PATH_LIST_OUT"]).read_text())
total = 0
for e in entries:
    p = Path(e)
    total += p.stat().st_size
    sc = p.with_suffix(".json")
    if sc.is_file():
        total += sc.stat().st_size
print(f"input bytes for this slice: {total}")
PY
```

Record this as `input_bytes_phase1`.

### 3.3 Pack the slice, serially, into scratch

```bash
uv run python -m diffusion_planner.data_pipeline.validation.pack_bench \
  --source "$SOURCE" \
  --dest-root "$SCRATCH/calib/phase1" \
  --workers 1 \
  --path-list "$SLICE_PATH_LIST_OUT" \
  --partition-regex "$PARTITION_REGEX" \
  --json-out "$SCRATCH/calib/phase1.json" \
  | tee "$SCRATCH/calib/phase1.log"
```

This prints a small table (`workers samples wall_s samples/s eff written_MB rss_self_MB
rss_children_MB`) and, per partition, one line from the underlying pack call itself
(`<partition id>: <N> kept, <N> rejected, <N> missing sidecars, built, <N> shards,
data_rev=... meta_rev=...`) — keep that log; Phase 2's determinism check needs it.

Two things this run's `rss_self`/`rss_children` numbers are **not**: they are cumulative
watermarks over the whole invocation, not a clean "this row's peak", and because this
invocation only measures one worker count they are, here, a reasonable reading of that single
config's memory footprint. They will stop being a clean reading the moment more than one worker
count is measured inside a single `pack_bench` call — which is exactly why every command in
this document runs `pack_bench` with one worker count at a time (see Phase 2 for the other
reason: reading the source tree at one worker count warms the page cache for whatever runs
next in the same call, which would quietly inflate a later row's throughput and make a
"cold" comparison meaningless).

From the JSON output, take `samples_per_s` as `r1`, and `bytes_written` to compute the ratio:

```bash
python -c "
import json
row = json.load(open('$SCRATCH/calib/phase1.json'))[0]
input_bytes = <input_bytes_phase1 from step 3.2>
print('r1 =', row['samples_per_s'], 'samples/s')
print('ratio =', row['bytes_written'] / input_bytes)
"
```

Record `r1` and the ratio. Label the cache state **approximate, not cold** — this host's page
cache cannot be dropped without root, so "previously unread by this procedure" is the best
approximation available, not a guarantee of a genuinely cold read.

---

## 4. Phase 2 — parallel calibration

Goal: measure scaling efficiency at a real worker count, confirm determinism between the serial
and parallel paths, and measure every additional worker count you are actually considering for
production — each one independently, with its **own** efficiency, because a measurement at one
worker count does not justify using a different one, and its efficiency does not carry over to
a different one either.

### 4.1 A fresh, equally unread, equally stratified slice, at a real worker count

```bash
export SLICE_TARGET_SAMPLES="$CALIB_SAMPLES"
export SLICE_PARTITIONS_OUT="$SCRATCH/phase2_fresh.partitions.txt"
export SLICE_PATH_LIST_OUT="$SCRATCH/phase2_fresh.path_list.json"
uv run python "$SCRATCH/select_slice.py"    # draws only partitions not already in $USED_PARTITIONS

uv run python -m diffusion_planner.data_pipeline.validation.pack_bench \
  --source "$SOURCE" \
  --dest-root "$SCRATCH/calib/phase2_fresh" \
  --workers 8 \
  --path-list "$SLICE_PATH_LIST_OUT" \
  --partition-regex "$PARTITION_REGEX" \
  --json-out "$SCRATCH/calib/phase2_fresh.json" \
  | tee "$SCRATCH/calib/phase2_fresh.log"
```

Because this is a *different* slice from Phase 1's, matched only by construction (same slicing
method, same target size), efficiency here is not confounded by Phase 1 having already warmed
the page cache for these particular files. Efficiency is **per worker count**, not a single
corpus-wide constant — compute it for this row as:

```
η_8 = samples_per_s(this row) / (r1 × 8)
```

Record `η_8` (do not just call it `η`; §4.3 measures a distinct efficiency for every other
candidate worker count, and §5.4 needs to know which one belongs to the count you actually
chose), and this run's `rss_self`/`rss_children` as the parallel-run memory reading, together
with whatever read-throughput and other-jobs'-throughput readings your named observer collects
during this run — this is the run that reading §0's host-safety agreement calls for a human to
be watching.

### 4.2 Determinism: a warm re-pack of Phase 1's exact slice, at the same worker count

```bash
uv run python -m diffusion_planner.data_pipeline.validation.pack_bench \
  --source "$SOURCE" \
  --dest-root "$SCRATCH/calib/phase1_repack_w8" \
  --workers 8 \
  --path-list "$SCRATCH/phase1.path_list.json" \
  --partition-regex "$PARTITION_REGEX" \
  --json-out "$SCRATCH/calib/phase1_repack_w8.json" \
  | tee "$SCRATCH/calib/phase1_repack_w8.log"
```

This one is deliberately warm (it reads the same files Phase 1 just read) — it is not a
throughput measurement, it exists purely to check that packing the same samples at a different
worker count produces the identical result. Compare the two logs:

```bash
grep -oE '^[^:]+: .*data_rev=[a-f0-9]+ meta_rev=[a-f0-9]+' "$SCRATCH/calib/phase1.log" \
  | sort > "$SCRATCH/calib/phase1_revs.txt"
grep -oE '^[^:]+: .*data_rev=[a-f0-9]+ meta_rev=[a-f0-9]+' "$SCRATCH/calib/phase1_repack_w8.log" \
  | sort > "$SCRATCH/calib/phase1_repack_w8_revs.txt"
diff "$SCRATCH/calib/phase1_revs.txt" "$SCRATCH/calib/phase1_repack_w8_revs.txt"
```

`diff` must report no differences: identical `data_rev` and `meta_rev` per partition, at two
different worker counts. Then confirm the shard bytes themselves match, not just their recorded
revisions:

```bash
find "$SCRATCH/calib/phase1/w1/shards" -name 'shard-*.tar' -exec sha256sum {} \; \
  | awk '{print $1, $2}' | sed 's#.*/shards/##' | sort > "$SCRATCH/calib/phase1_shard_hashes.txt"
find "$SCRATCH/calib/phase1_repack_w8/w8/shards" -name 'shard-*.tar' -exec sha256sum {} \; \
  | awk '{print $1, $2}' | sed 's#.*/shards/##' | sort > "$SCRATCH/calib/phase1_repack_shard_hashes.txt"
diff "$SCRATCH/calib/phase1_shard_hashes.txt" "$SCRATCH/calib/phase1_repack_shard_hashes.txt"
```

If either `diff` shows a difference, stop. Do not proceed to a production pack until you have
found and fixed the cause — worker count must never change the bytes a partition produces.

### 4.3 Every other worker count under consideration

Repeat 4.1 (a fresh slice, then `pack_bench` at that worker count) once per remaining value in
`$CANDIDATE_WORKERS`, each into its own scratch dest-root:

```bash
for W in $CANDIDATE_WORKERS; do
  [ "$W" = 8 ] && continue   # already measured in 4.1
  export SLICE_TARGET_SAMPLES="$CALIB_SAMPLES"
  export SLICE_PARTITIONS_OUT="$SCRATCH/phase2_w${W}.partitions.txt"
  export SLICE_PATH_LIST_OUT="$SCRATCH/phase2_w${W}.path_list.json"
  uv run python "$SCRATCH/select_slice.py"

  uv run python -m diffusion_planner.data_pipeline.validation.pack_bench \
    --source "$SOURCE" \
    --dest-root "$SCRATCH/calib/phase2_w${W}" \
    --workers "$W" \
    --path-list "$SLICE_PATH_LIST_OUT" \
    --partition-regex "$PARTITION_REGEX" \
    --json-out "$SCRATCH/calib/phase2_w${W}.json" \
    | tee "$SCRATCH/calib/phase2_w${W}.log"
done
```

For each candidate, compute its own efficiency the same way §4.1 did — `η` is a function of
worker count, not one number that carries over from whichever count you happened to measure
first:

```
η_W = samples_per_s(that candidate's row) / (r1 × W)
```

Record `samples_per_s`, `rss_self`/`rss_children`, and this `η_W` for every candidate, including
the `W=8` row already recorded as `η_8` in §4.1 — you should end this step with one `η_W` per
entry in `$CANDIDATE_WORKERS`, not one shared value. A worker count you did not measure this way
is not a worker count you are entitled to use in Phase 3 — "it should scale further" is not a
measurement, and neither is reusing another count's efficiency for it.

Once every candidate is measured, choose `$WORKERS` (the value you will use for the production
pack) from the evidence, set it, and record which candidate's `η_W` that decision carries
forward — that specific number, not "the Phase 2 efficiency," is what §5.4 projects with:

```bash
export WORKERS=<the worker count Phase 2 actually justified>
export ETA=<the η_W measured for that exact worker count, from this step or from §4.1>
```

---

## 5. Phase 3 — batched production pack

Not one invocation — see trap 2 in the introduction. Pack in batches of roughly `$BATCH_SIZE`
partitions, each batch a `pack` call selecting its own partitions with repeated `--partition`
flags, chaining `--base` to the previous batch's tag (`none` for the very first batch).

### 5.1 Build expected keys for later acceptance, once, now

Before packing anything, turn the rewritten path list into the source-relative key list that
`verify_conversion` will need in Phase 4 — a "key" is a path-list entry with the source prefix
and the trailing `.npz` removed:

```bash
uv run python - <<PY
import json
from pathlib import Path

source = Path("$SOURCE").resolve()
entries = json.loads(Path("$PATH_LIST").read_text())
keys = [Path(e).resolve().relative_to(source).as_posix()[:-4] for e in entries]
Path("$SCRATCH/expected_keys.json").write_text(json.dumps(keys))
print(len(keys), "expected keys written")
PY
```

Keep in mind while reading Phase 4 later: this expected-keys file is *every* entry in the path
list. If any samples end up dropped as `is_skipped` during packing (the default `pack` behaviour
drops them; nothing above disables that default), those keys will be in this file but not in the
published manifest, and the membership check in Phase 4 will report them as missing rather than
silently accepting them — it does not know on its own that a dropped-skip is an allowed
difference. Track the "rejected" counts each batch prints (§5.3) as you go; if Phase 4's missing
count does not match the sum of those, that is a real problem, not an allowed one.

### 5.2 Split the corpus into batches

```bash
uv run python - <<PY
import os
from pathlib import Path

rows = [l.split("\t") for l in Path(os.environ["PARTITIONS_TSV"]).read_text().splitlines()]
pids = [pid for pid, _ in rows]
batch_size = int(os.environ["BATCH_SIZE"])
outdir = Path(os.environ["SCRATCH"]) / "batches"
outdir.mkdir(parents=True, exist_ok=True)
for i in range(0, len(pids), batch_size):
    (outdir / f"batch_{i // batch_size:04d}.txt").write_text(
        "\n".join(pids[i : i + batch_size]) + "\n"
    )
print(f"{(len(pids) + batch_size - 1) // batch_size} batches written to {outdir}")
PY
```

### 5.3 Pack each batch, chaining `--base`

```bash
prev_tag=none
i=0
for batch in "$SCRATCH"/batches/batch_*.txt; do
  tag="${TAG_PREFIX}-batch-$(printf '%04d' "$i")"
  args=()
  while IFS= read -r pid; do
    [ -n "$pid" ] && args+=(--partition "$pid")
  done < "$batch"

  uv run python -m diffusion_planner.data_pipeline.pack_shards pack \
    --source "$SOURCE" \
    --dest "$DEST" \
    --base "$prev_tag" \
    --tag "$tag" \
    --partition-regex "$PARTITION_REGEX" \
    --path-list "$PATH_LIST" \
    --workers "$WORKERS" \
    --require-sidecars \
    "${args[@]}" \
    2>&1 | tee "$SCRATCH/batch_${i}.log"

  prev_tag="$tag"
  i=$((i + 1))
done
echo "final batch tag: $prev_tag"
```

`--require-sidecars` makes a batch with any sidecar-less sample fail before it publishes anything
— this is the sidecar gate, and it only protects you if every batch actually carries the flag.

**Important:** every successful `pack` call also updates the dataset's `latest` pointer to that
call's own tag — including every intermediate batch tag, which is by construction an incomplete
dataset (it only has the partitions packed so far). Nothing that reads `latest` during Phase 3
should be pointed at this dataset yet. The tag you hand to training is the *final* batch's tag,
and only after it passes Phase 4 in full.

### 5.4 Re-measure against your calibration projection

After batch 0 finishes, and again once roughly 5% of the corpus's total samples have been
packed, compare actual progress against Phase 1/2's cost model. The model: wall time for `S`
samples at `W` workers is `S / (r1 × W × η_W)`, using `r1` from Phase 1 and, critically, the
`η_W` that §4.3 measured **for the specific `W` you set as `$WORKERS`** — not `η_8` unless
`$WORKERS` is actually `8`. §4.3 measured a different efficiency for every candidate precisely
so that a projection could not silently reuse the wrong one; use `$ETA` as recorded there. This
is also the number to use if anyone asks how long a full rebuild costs.

```bash
python -c "
samples_so_far = <sum of 'kept' counts from the batch logs so far>
elapsed_so_far = <wall-clock seconds since batch 0 started>
r1 = <from Phase 1>
eta = <\$ETA — the efficiency measured for \$WORKERS specifically, from §4.1 or §4.3>
workers = <\$WORKERS>
observed_rate = samples_so_far / elapsed_so_far
expected_rate = r1 * workers * eta
print('observed samples/s:', observed_rate)
print('expected samples/s:', expected_rate)
print('ratio:', observed_rate / expected_rate)
"
```

If the observed rate is meaningfully below the expected rate (a large gap — for illustration,
more than a 20-30% shortfall — is worth pausing over; use your own judgement for what "meaningful"
means on this host), stop and investigate before continuing to spend the remaining batches. This
is the entire point of not packing as one invocation: the course correction has to happen while
there is still a run left to correct.

---

## 6. Phase 4 — acceptance

No dataset is handed to training before every one of the following passes and is written down.
Use `$FINAL_TAG` = the last batch's tag from Phase 3.

### 6.1 Scrub

Full payload re-hash of every shard against its manifest:

```bash
uv run python -m diffusion_planner.data_pipeline.pack_shards scrub \
  --dest "$DEST" --tag "$FINAL_TAG"
```

Expect `scrub OK: <members> members in <shards> shards, 0 mismatches`. `scrub` does not check
member *offsets* — only payload hash and size — which is what the next step is for.

### 6.2 Offset integrity and two-way membership

```bash
uv run python -m diffusion_planner.data_pipeline.validation.verify_conversion \
  --dest "$DEST" \
  --tag "$FINAL_TAG" \
  --expected-keys-json "$SCRATCH/expected_keys.json"
```

On success this prints `offsets OK: {...}` then `membership OK: {...}`. On failure it prints
`error: ...` and exits non-zero. If it fails on membership with a nonzero `missing_from_manifest`,
compare that count against the total "rejected" count you tracked across every batch's log in
§5.1 — an exact match means the difference is entirely accounted for by `is_skipped` drops (a
sanctioned difference; record the count and move on), and anything else means a real integrity
problem that must be resolved before continuing.

### 6.3 Seeded, per-source bit-exact sample of at least `$MIN_BITEXACT_KEYS` keys

There is no dedicated bit-exact-sampling command; build the sample from `pack_shards export`,
which decodes matching samples straight out of the shards and re-materializes them as `.npz` +
sidecar files for comparison. Its `--where` argument is a single SQL boolean expression (no
`ORDER BY`, no `LIMIT`, and no `;`), so the seeded, per-source selection has to be expressed as a
deterministic hash-based filter rather than `ORDER BY random() LIMIT N`.

First, compute a per-source threshold that comfortably clears your quota:

```bash
export PARTITIONS_TSV="$SCRATCH/partitions.tsv"
uv run python - <<PY > "$SCRATCH/bitexact_where.tsv"
import math, os
from pathlib import Path

totals = {}
for line in Path(os.environ["PARTITIONS_TSV"]).read_text().splitlines():
    pid, n = line.split("\t")
    top = pid.split("/", 1)[0]
    totals[top] = totals.get(top, 0) + int(n)

quota = math.ceil(int("$MIN_BITEXACT_KEYS") / len(totals))
for src, total in sorted(totals.items()):
    threshold = min(1000, math.ceil(quota / total * 1000)) if total else 1000
    print(f"{src}\t{threshold}")
PY
```

Then export one sample per source:

```bash
mkdir -p "$SCRATCH/bitexact_sample"
while IFS=$'\t' read -r SRC THRESHOLD; do
  uv run python -m diffusion_planner.data_pipeline.pack_shards export \
    --dest "$DEST" \
    --tag "$FINAL_TAG" \
    --where "(partition_id = '$SRC' OR partition_id LIKE '$SRC/%') AND (hash(key || '_$SAMPLE_SEED') % 1000) < $THRESHOLD" \
    --out "$SCRATCH/bitexact_sample/$SRC"
done < "$SCRATCH/bitexact_where.tsv"
```

Each export writes an `export_manifest.json` under its own directory listing exactly which keys
it drew. Confirm the combined total meets `$MIN_BITEXACT_KEYS`; if it falls short, raise the
per-source thresholds and re-run (raising a threshold only ever adds keys to the same
deterministic selection, so this is safe to re-run into the same output directories).

Now compare every exported array against the original source file, at the array level — not by
diffing the `.npz` files themselves, since re-serializing arrays into a fresh `.npz` container
does not reproduce the original file's bytes even when every array inside is identical:

```bash
uv run python - <<PY
import json
from pathlib import Path

import numpy as np

out_root = Path("$SCRATCH/bitexact_sample")
source = Path("$SOURCE")
checked = mismatches = 0
for manifest_path in out_root.rglob("export_manifest.json"):
    keys = json.loads(manifest_path.read_text())["keys"]
    d = manifest_path.parent
    for key in keys:
        checked += 1
        got = np.load(d / f"{key}.npz")
        want = np.load(source / f"{key}.npz")
        if set(got.files) != set(want.files):
            print(f"MISMATCH {key}: array names differ")
            mismatches += 1
            continue
        for name in got.files:
            a, b = got[name], want[name]
            if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
                print(f"MISMATCH {key}/{name}")
                mismatches += 1
                break
print(f"checked {checked} keys, {mismatches} mismatch(es)")
assert checked >= int("$MIN_BITEXACT_KEYS"), "sample fell short of the floor"
assert mismatches == 0
PY
```

### 6.4 Loader smoke

Build a keyset over the final version, then run the same loader bench used for the shard-vs-npz
throughput comparison, from an environment that has `torch` installed:

```bash
uv run python -m diffusion_planner.data_pipeline.pack_shards keyset \
  --dest "$DEST" --tag "$FINAL_TAG" \
  --where "true" \
  --out "$SCRATCH/final_keyset.parquet"

uv run python -m diffusion_planner.data_pipeline.validation.throughput_bench \
  --dataset-root "$DEST" \
  --version "$FINAL_TAG" \
  --keyset "$SCRATCH/final_keyset.parquet" \
  --steps 200
```

Confirm it runs to completion with no mapping faults. This is a limited claim by design — a
short run over a handful of steps cannot exercise every mapping in a large corpus; that is what
§6.2's offset/membership check already covers at full scale. If you also want to record the
loader's expected `steps_per_epoch` for this version, read it directly off the dataset it builds
internally (this uses the same production classes the bench above already imports, just reading
one more property off the object instead of only its throughput):

```bash
uv run python - <<PY
from diffusion_planner.utils.shard_dataset import ShardDatasetConfig, make_shard_dataloader
from pathlib import Path

cfg = ShardDatasetConfig(
    root=Path("$DEST"),
    version="$FINAL_TAG",
    keyset_path=Path("$SCRATCH/final_keyset.parquet"),
    batch_size=64,
    world_size=1,
    rank=0,
    num_workers=0,
    seed=0,
)
loader = make_shard_dataloader(cfg, pin_memory=False)
print("steps_per_epoch:", loader.dataset.steps_per_epoch)
PY
```

Compare that number against what you expect from the total kept-sample count (§6.2's `members`)
divided by `batch_size × world_size`, and record both.

### 6.5 Cost record

Write down, in one place, for the record:

- `r1` (Phase 1), the production worker count `$WORKERS`, and the `η_W` that specific worker
  count measured in Phase 2 (`$ETA` — not any other candidate's efficiency);
- the observed production rate: total kept samples across all batches ÷ total wall-clock time
  across all batches;
- total wall-clock time for the whole production pack;
- total bytes written — measure this directly (`du -sb` over the shard/manifest directories for
  `$FINAL_TAG`'s partitions), do not rely on the Phase 1 ratio for the real figure;
- total bytes read — the packer reads every kept sample twice plus its sidecar once plus a small
  re-check pass; if you have no direct host I/O accounting, report this as an estimate
  (`≈ 2 × kept-sample bytes + sidecar bytes`) and label it as such, the same way you would label
  any other number you did not measure directly;
- peak coordinator RSS and peak per-worker RSS from Phase 2's measurements (`rss_self`,
  `rss_children`), labelled as bench-invocation cumulative watermarks, not a clean per-run peak;
- the §6.1–6.4 pass/fail results and the exact sample and mismatch counts from each.

This record, not this document, is what should be handed to anyone asking "how long would it
take to rebuild this, and how do we know it's correct."

---

## 7. Abort

Any one of the following stops the run immediately — these are the conditions actually specified
for this kind of run, reproduced exactly rather than padded to a round number:

1. Anyone responsible for another job on the shared filesystem reports a throughput impact.
2. The shared filesystem's free space falls below `$FREE_SPACE_FLOOR_GB`.
3. Any `SourceChangedError` (a file changed between being read for hashing and being read for
   encoding — the source tree was supposed to be immutable for the duration; if this fires, that
   assumption was violated and needs to be re-established before anything continues).
4. Any `BrokenProcessPool`, or any worker death that is not a clean, expected data error. The CLI
   surfaces a `BrokenProcessPool` as an `error: a pack worker died without raising (check the OOM
   killer and dmesg); partitions in flight: [...]` message, not a raw traceback — if you see a raw
   traceback instead, something is more wrong than the tool anticipated and that is its own reason
   to stop.

**Emergency stop.** Killing only the coordinator process is not sufficient — worker processes are
spawned (not forked), so they are independent OS processes that will keep running. Terminate the
whole process group:

```bash
kill -TERM -- -"$(ps -o pgid= -p <coordinator pid> | tr -d ' ')"
# escalate if needed:
kill -KILL -- -"$(ps -o pgid= -p <coordinator pid> | tr -d ' ')"
```

Then confirm no stale writer lock remains on `$DEST`. The lock is an advisory OS file lock tied
to a file descriptor, released automatically the moment every process holding it has actually
exited — so this is a check that the kill above actually finished, not a separate cleanup step:

```bash
uv run python - <<PY
import fcntl, os

path = "$DEST/.writer.lock"
fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o664)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(fd, fcntl.LOCK_UN)
    print("lock is free")
except BlockingIOError:
    print("still held — a process from the killed run is still alive; find and stop it")
finally:
    os.close(fd)
PY
```

If it reports still held, some process from the run survived the kill (check for a wedged or
reparented process rather than assuming the lock file itself needs deleting — deleting it while
something still holds the underlying descriptor does not release that hold). Only resume packing
once this reports free.
