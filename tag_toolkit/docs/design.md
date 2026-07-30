# TagStore: design notes

What `TagStore` actually does under the hood, and the contracts that aren't
obvious from the API. For method signatures and examples, see
[`usage.md`](usage.md).

---

## Source (the authoritative frame set)

A TagStore is built from a `source` at construction time and that set is
**fixed for the lifetime of the instance**: the constructor scans, the index
is built, and subsequent calls only ever operate on frames that were present
in that scan. There is no way to add frames later; if the data on disk
changes, rebuild the index.

`clause`, `scope`, and `frame_filter` are **only filters** — they narrow the
set, they never extend it. A path that isn't in the index never becomes
"visible" through `scope`.

`source` accepts any shape that [`expand_source`](../source.py) understands
(directory, path-list JSON, single NPZ, list of those, or an existing
`.tag` index file):

|  Form  |  Example  |
| --- | --- |
|  Index file  |  `TagStore("/path/to/tags.tag")`  |
|  Directory  |  `TagStore("/path/to/dataset")`  |
|  Path-list JSON  |  `TagStore("/path/to/list.json")`  |
|  List of paths  |  `TagStore([path1, path2])`  |
|  Single NPZ  |  `TagStore("/path/to/frame.npz")`  |
|  None (empty)  |  `TagStore()`  |

For large or repeated use, prefer a pre-built `.tag` index file. The on-init
directory scan reads every sidecar on disk; the loaded index skips that.

---

## Atomic writes

A mutation (`add_tags`, `remove_tags`, `remove_dimension`, `replace_tags`,
`add_tags_to_route`) writes to one sidecar at a time via a `.json.tmp` sibling that
gets `os.replace`d over the target on success. If anything fails, the
original `.json` is left untouched and the temp file is removed. A partial
mutation can therefore leave the dataset in a half-tagged state across
frames — but never corrupts a single sidecar.

The atomic-rename pattern does not protect against power loss between
`write(2)` and the parent-directory entry being durable on disk; if a
crash lands inside that gap, the sidecar may be missing despite the
caller seeing success. For dataset publishing pipelines that demand
stronger durability, run `fsync(f.fileno())` plus an `fsync` on the
parent directory in the same critical section — out of scope here, but
worth knowing before relying on writes through a network filesystem.

---

## Scope resolution

`scope` looks informal from the outside but the resolution has a deliberate
order: fast paths first, disk fallback last, and a strict never-extends
guarantee. The contract is the same for query and mutate.

`_resolve_scope(scope, granularity)` returns a `set[Path]` of either route
directories (`granularity="route"`) or NPZ paths (`granularity="frame"`).
Three cases, evaluated in order:

1. **`scope is None`** — return every path in the index at the requested
   granularity. No filesystem work.
2. **`scope` is a `TagStore`** — return every path from that store's index
   at the requested granularity, then **intersect** with this store's index.
   Routes from the scope store that don't exist here drop out silently.
   The intersection is what makes nested queries behave identically to
   passing in a list of paths: a route only contributes if it also has
   frames in this store.
3. **`scope` is a Path or list of Paths** — for each item, try in order:
   - **route path already in the index** — O(1) lookup in
     `frames_of_route`. Return `{route}` for route granularity, or the
     route's frame list for frame granularity.
   - **NPZ path already in the index** — O(1) lookup in `route_of_frame`.
     Return `{route}` for route granularity, or `{npz}` for frame.
   - **fallback** — for anything else (a directory, a path-list JSON, a
     typo'd path, anything not in the index), use
     [`expand_source`](../source.py) to enumerate NPZ files on disk, then
     keep only the ones that are also in the index.

Anything that ends up with no match in the index is silently dropped. The
fallback is the only place that touches disk during a `scope` resolution, so
passing already-known routes or NPZs (cases 3a/3b) avoids it entirely on
repeated operations.

`add_tags_to_route` is a deliberate exception: it rejects with `ValueError` when
the route isn't in the index instead of falling back, because silently
tagging frames that aren't indexed would make those tags invisible to
every subsequent query — a footgun rather than a feature.

---

## Source vs Scope

`source` and `scope` look similar but are deliberately distinct roles:

| | `source` (constructor) | `scope` (per-call) |
|---|---|---|
| When set | Once, at `TagStore(source)` | Every query / mutate call |
| Effect | Defines the **authoritative frame set** — what the store ever knows about | Narrows the **operation** to a subset of the store's frames |
| Can extend the frame set? | Yes — scanning discovers frames on disk | **No** — unknown paths are silently dropped (or raise `FileNotFoundError` if they look like a directory) |
| Disk cost | One-time scan or `.tag` load | None for indexed paths; one `expand_source` fallback per non-indexed path |
| Mutability | Frozen for the lifetime of the store | Re-evaluated per call |

`source` answers "what does this store see?". `scope` answers "of what
this store sees, which subset does this call care about?". Confusing them
is the most common source of bugs:

- Passing a `Path` that doesn't exist to `TagStore(source)` raises
  `FileNotFoundError` — that's the store refusing to fabricate an
  authoritative set.
- Passing the same `Path` to `query(..., scope=path)` silently drops it —
  the store already has its authoritative set; the path just didn't match
  any of it.

Treat them as different categories. `source` is a build-time contract;
`scope` is a runtime filter.
