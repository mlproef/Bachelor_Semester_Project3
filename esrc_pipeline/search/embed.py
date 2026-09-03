"""Embed text for Search (BSP-compatible default: all-mpnet-base-v2)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from search.paths import DEFAULT_EMBED_MODEL


def embed_texts(
    texts: Sequence[str],
    *,
    model_name: str = DEFAULT_EMBED_MODEL,
) -> np.ndarray:
    """Return (N, D) float32 embedding matrix.

    TODO — optional: OpenAI-compatible remote embed (e.g. jina on gpu6)
    when local sentence-transformers is unavailable.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "sentence-transformers required: pip install sentence-transformers"
        ) from e

    model = SentenceTransformer(model_name)
    vecs = model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)
