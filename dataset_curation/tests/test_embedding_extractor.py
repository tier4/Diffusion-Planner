from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


def test_pool_embedding_mean():
    from dataset_curation.embedding_extractor import _pool_embedding

    enc = torch.randn(1, 50, 256)
    pooled = _pool_embedding(enc)
    assert pooled.shape == (256,)
    expected = enc[0].mean(dim=0)
    torch.testing.assert_close(pooled, expected)


def test_pool_embedding_batch():
    from dataset_curation.embedding_extractor import _pool_embedding

    enc = torch.randn(4, 50, 256)
    pooled = _pool_embedding(enc)
    assert pooled.shape == (4, 256)
