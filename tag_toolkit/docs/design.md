# TagStore: design notes

What `TagStore` does under the hood, and the contracts that aren't obvious
from the API. For method signatures and examples see
[`usage.md`](usage.md).

---

## Source (the authoritative frame set)

A TagStore is built from a `source` at construction time and that set is
**fixed for the lifetime of the instance**. The constructor scans, the index
is built, and subsequent calls only ever operate on frames that were
present in that scan. There is no way to add frames later; if the data on
disk changes, rebuild the index.

`clause`, `scope`, and `frame_filter` are **only filters** — they narrow
the set, they never extend it. A path that isn't in the index never
becomes "visible" through `scope`.

For large or repeated use, prefer a pre-built `.tag` index file. The
on-init directory scan reads every sidecar on disk; the loaded index
skips that.

---

## Atomic writes

A mutation writes to one sidecar at a time via a `.json.tmp` sibling
that gets `os.replace`d over the target on success. If anything fails, the
original `.json` is left untouched and the temp file is removed. A partial
mutation can leave the dataset in a half-tagged state across frames — but
never corrupts a single sidecar.

### Verify-then-write

Every mutation runs a stale-check just before the atomic write. The check
piggy-backs on the `read_sidecar` call that `write_tags` already makes to
preserve non-tag fields: after reading the sidecar, it compares the
on-disk `tags` field against an `expected_tags` value the store passes in
(`index.frame_tags[npz]` — the index's view of what the frame had).

- **On-disk matches the index** — the write proceeds.
- **On-disk has drifted** — `StaleIndexError` is raised, the sidecar is
  left untouched, and the mutation is aborted. No further frames are
  touched in this call.
- **No `expected_tags` passed** — the check is skipped. This is for
  out-of-band tools without an index to compare against. The store's
  mutation methods always pass it.

`StaleIndexError` is the recovery signal. After fixing the drift
(manually, or via `reindex_tags()` to refresh the index from disk), retry
the mutation:

```python
try:
    store.add_tags(["site:new"], scope=route)
except StaleIndexError:
    store.reindex_tags()
    store.add_tags(["site:new"], scope=route)
```

`write_tags` will not create a sidecar that didn't already exist on disk.
A frame without a sidecar raises `FileNotFoundError` from the mutation
preflight.

### fsync

Each write by default calls `fsync` on the file and on the parent
directory. The directory fsync ensures the filename → inode mapping is
persisted; without it, a power loss can leave the file content on disk
but the filename missing from the directory.

fsync is expensive (most SSDs handle only 500–2000 fsyncs/s). For batch
workloads, use `mutation_scope` (next section) to defer fsync to scope
exit.

### `mutation_scope` defers fsync

`mutation_scope` defers only fsync. Per-write file writes and in-memory
index updates happen immediately; the directory-level fsync is performed
once at scope exit (one per touched directory).

The smart default: `sync=None` on every mutating method means
"follow the surrounding scope". Outside a `mutation_scope`, `sync=None`
resolves to `True` (per-file fsync, durable). Inside a `mutation_scope`,
it automatically becomes `False` (no per-file fsync; batched at exit).
Explicit `sync=True`/`False` overrides both the default and the scope.

```python
with store.mutation_scope():
    store.add_tags(["site:foo"], scope=route1)
    store.add_tags(["site:bar"], scope=route2)
# fsync happens once per directory here
```

The `sync` argument on `mutation_scope` itself (`True` by default)
controls whether the scope performs directory-level fsync at all. Pass
`False` to skip fsync entirely (useful inside larger pipelines that
manage their own durability).

---

## Scope resolution

`_resolve_scope(scope, granularity)` returns a `set[Path]` of either
route directories (`granularity="route"`) or NPZ paths
(`granularity="frame"`). Three cases, in order:

1. **`scope is None`** — return every path in the index at the requested
   granularity. No filesystem work.
2. **`scope` is a `TagStore`** — return every path from that store's
   index, then **intersect** with this store's index. Routes from the
   scope store that don't exist here drop out silently. The intersection
   is what makes nested queries behave identically to passing a list of
   paths: a route only contributes if it also has frames in this store.
3. **`scope` is a Path or list of Paths** — for each item, try in order:
   - **route path already in the index** — O(1) lookup in
     `frames_of_route`. Return `{route}` (route granularity) or the
     route's frame list (frame granularity).
   - **NPZ path already in the index** — O(1) lookup in
     `route_of_frame`. Return `{route}` (route granularity) or `{npz}`
     (frame).
   - **fallback** — for anything else (directory, path-list JSON,
     unknown path), use `expand_source` to enumerate NPZ files on disk,
     then keep only the ones that are also in the index.

Anything that ends up with no match in the index is silently dropped. The
fallback is the only place that touches disk during scope resolution, so
passing already-known routes or NPZs (cases 3a/3b) avoids it entirely.

`add_tags_to_route` is a deliberate exception: it rejects with
`ValueError` when the route isn't in the index, because silently tagging
frames that aren't indexed would make those tags invisible to every
subsequent query.

---

## Source vs Scope

`source` and `scope` look similar but are deliberately distinct:

| | `source` (constructor) | `scope` (per-call) |
|---|---|---|
| When set | Once, at `TagStore(source)` | Every query / mutate call |
| Effect | Defines the **authoritative frame set** | Narrows the **operation** to a subset |
| Can extend the frame set? | Yes — scanning discovers frames on disk | **No** — unknown paths are silently dropped |
| Disk cost | One-time scan or `.tag` load | None for indexed paths; one `expand_source` fallback per non-indexed path |
| Mutability | Frozen for the lifetime of the store | Re-evaluated per call |

`source` answers "what does this store see?". `scope` answers "of what
this store sees, which subset does this call care about?". Confusing
them is the most common source of bugs:

- Passing a `Path` that doesn't exist to `TagStore(source)` raises
  `FileNotFoundError` — the store refuses to fabricate an authoritative
  set.
- Passing the same `Path` to `query(..., scope=path)` silently drops it
  — the store already has its authoritative set; the path just didn't
  match any of it.

Treat them as different categories. `source` is a build-time contract;
`scope` is a runtime filter.