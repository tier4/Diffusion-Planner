from pathlib import Path


def data_path_to_rel(path) -> Path:
    """Map an input data ``.npz`` path to a RELATIVE output path that mirrors the input
    directory hierarchy below the dataset split anchor (``valid`` or ``train``), with the
    extension dropped.

    Saving under ``<out_dir>/<rel>.npz`` reproduces the input's ``<location>/<split>/
    <date>/<time>/<frame>`` structure. The path is unique per data point, so different
    ranks/GPUs never collide.
    """
    parts = Path(path).parts
    split_idx = next((i for i, p in enumerate(parts) if p in ("valid", "train")), len(parts) - 4)
    return Path(*parts[split_idx - 1 :]).with_suffix("")  # drop .npz
