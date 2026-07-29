"""Single source of truth for the ``file://`` rendezvous store used by DDP init.

The path must differ per user. /tmp carries the sticky bit, so a store file left behind by
another user cannot be unlinked, and the launchers' stale-store cleanup dies with
PermissionError before training even starts. Keying the file name on the uid keeps every
user on their own store.
"""

import os
from pathlib import Path


def dist_init_file_path() -> Path:
    """Rendezvous store shared by all ranks of one run (see ``ddp_setup_universal``)."""
    return Path(f"/tmp/tmp_dist_init_uid{os.getuid()}")
